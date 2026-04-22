// Copyright (c) Prevail Verifier contributors.
// SPDX-License-Identifier: Apache-2.0
#include <map>
#include <set>
#include <utility>
#include <variant>
#include <ranges>

#include "cfg/cfg.hpp"
#include "cfg/wto.hpp"
#include "config.hpp"
#include "crab/ebpf_domain.hpp"
#include "ir/program.hpp"
#include "result.hpp"
#include "verifier.hpp"

namespace prevail {

thread_local LazyAllocator<ProgramInfo> thread_local_program_info;
thread_local ebpf_verifier_options_t thread_local_options;

static void ebpf_verifier_clear_before_analysis() {
    clear_thread_local_state();
    variable_registry.clear();
}

void ebpf_verifier_clear_thread_local_state() {
    CrabStats::clear_thread_local_state();
    thread_local_program_info.clear();
    clear_thread_local_state();
    SplitDBM::clear_thread_local_state();
}

class InterleavedFwdFixpointIterator final {
    struct WorstInstrCount {
        uint64_t count{};
        bool unbounded{};
    };

    struct PathSummary {
        bool reaches_exit{};
        uint64_t count{};
        bool unbounded{};
    };

    const Program& _prog;
    const Cfg& _cfg;
    const Wto _wto;
    AnalysisResult& result;

    /// number of narrowing iterations. If the narrowing operator is
    /// indeed a narrowing operator this parameter is not
    /// needed. However, there are abstract domains for which an actual
    /// narrowing operation is not available so we must enforce
    /// termination.
    static constexpr unsigned int _descending_iterations = 2000000;

    /// Used to skip the analysis until _entry is found
    bool _skip{true};

    [[nodiscard]]
    bool has_error(const Label& node) const {
        return result.invariants.at(node).error.has_value();
    }

    void set_error(const Label& node, VerificationError&& error) {
        result.failed = true;
        result.invariants.at(node).error = std::move(error);
    }

    void set_pre(const Label& label, EbpfDomain&& v) { result.invariants.at(label).pre = std::move(v); }
    void set_pre(const Label& label, const EbpfDomain& v) { result.invariants.at(label).pre = v; }

    EbpfDomain get_pre(const Label& node) const { return result.invariants.at(node).pre; }

    EbpfDomain get_post(const Label& node) const { return result.invariants.at(node).post; }

    void transform_to_post(const Label& label, EbpfDomain pre) {
        const auto& ins = _prog.instruction_at(label);
        if (!std::holds_alternative<IncrementLoopCounter>(ins)) {
            if (has_error(label)) {
                return;
            }
            for (const auto& assertion : _prog.assertions_at(label)) {
                // Avoid redundant errors.
                if (auto error = ebpf_domain_check(pre, assertion, label)) {
                    set_error(label, std::move(*error));
                    return;
                }
            }
        }
        ebpf_domain_transform(pre, ins);

        result.invariants.at(label).post = std::move(pre);
    }

    EbpfDomain join_all_prevs(const Label& node) const {
        if (node == _cfg.entry_label()) {
            return get_pre(node);
        }
        EbpfDomain res = EbpfDomain::bottom();
        for (const Label& prev : _cfg.parents_of(node)) {
            res |= get_post(prev);
        }
        return res;
    }

    explicit InterleavedFwdFixpointIterator(const Program& prog, AnalysisResult& result)
        : _prog(prog), _cfg(prog.cfg()), _wto(prog.cfg()), result(result) {
        for (const auto& label : _cfg.labels()) {
            result.invariants.emplace(label, InvariantMapPair{EbpfDomain::bottom(), {}, EbpfDomain::bottom()});
        }
    }

    static std::optional<VerificationError> check_loop_bound(const Program& prog, const Label& label,
                                                             const EbpfDomain& pre) {
        if (std::holds_alternative<IncrementLoopCounter>(prog.instruction_at(label))) {
            const auto assertions = prog.assertions_at(label);
            if (assertions.size() != 1) {
                CRAB_ERROR("Expected exactly 1 assertion for IncrementLoopCounter");
            }
            return ebpf_domain_check(pre, assertions.front(), label);
        }
        return {};
    }

    void find_termination_errors(const Program& prog) {
        for (const auto& [label, inv_pair] : result.invariants) {
            if (inv_pair.pre.is_bottom()) {
                continue;
            }
            if (auto error = check_loop_bound(prog, label, inv_pair.pre)) {
                set_error(label, std::move(*error));
            }
        }
    }

    int max_loop_count() const {
        ExtendedNumber loop_count{0};
        // Gather the upper bound of loop counts from post-invariants.
        for (const auto& inv_pair : std::views::values(result.invariants)) {
            loop_count = std::max(loop_count, inv_pair.post.get_loop_count_upper_bound());
        }
        const auto m = loop_count.number();
        if (m && m->fits<int32_t>()) {
            return m->cast_to<int32_t>();
        }
        return std::numeric_limits<int>::max();
    }

    [[nodiscard]]
    static bool is_counted_instruction(const Instruction& ins) {
        return !std::holds_alternative<Undefined>(ins) && !std::holds_alternative<Assume>(ins) &&
               !std::holds_alternative<IncrementLoopCounter>(ins);
    }

    [[nodiscard]]
    static WorstInstrCount unknown_unbounded() {
        return {.count = 0, .unbounded = true};
    }

    [[nodiscard]]
    static WorstInstrCount overflow_unbounded() {
        return {.count = std::numeric_limits<uint64_t>::max(), .unbounded = true};
    }

    [[nodiscard]]
    static WorstInstrCount combine_unbounded(const WorstInstrCount& left, const WorstInstrCount& right) {
        if (left.count == std::numeric_limits<uint64_t>::max() || right.count == std::numeric_limits<uint64_t>::max()) {
            return overflow_unbounded();
        }
        return unknown_unbounded();
    }

    [[nodiscard]]
    static bool checked_add(const uint64_t left, const uint64_t right, uint64_t& out) {
        if (std::numeric_limits<uint64_t>::max() - left < right) {
            return false;
        }
        out = left + right;
        return true;
    }

    [[nodiscard]]
    static bool checked_mul(const uint64_t left, const uint64_t right, uint64_t& out) {
        if (left != 0 && std::numeric_limits<uint64_t>::max() / left < right) {
            return false;
        }
        out = left * right;
        return true;
    }

    static void collect_labels(const CycleOrLabel& component, std::vector<Label>& labels) {
        if (const auto label = std::get_if<Label>(&component)) {
            labels.push_back(*label);
            return;
        }
        const auto& cycle = *std::get<std::shared_ptr<WtoCycle>>(component);
        for (const auto& sub_component : cycle) {
            collect_labels(sub_component, labels);
        }
    }

    [[nodiscard]]
    std::optional<uint64_t> get_loop_bound(const Label& head) const {
        ExtendedNumber loop_count{0};
        for (const auto& inv_pair : std::views::values(result.invariants)) {
            if (inv_pair.post.is_bottom()) {
                continue;
            }
            loop_count = std::max(loop_count, inv_pair.post.get_loop_count_upper_bound(head));
        }

        const auto count = loop_count.number();
        if (!count || !count->fits<uint64_t>()) {
            return {};
        }
        return count->cast_to<uint64_t>();
    }

    [[nodiscard]]
    WorstInstrCount summarize_component(const CycleOrLabel& component) const {
        if (const auto label = std::get_if<Label>(&component)) {
            return {.count = is_counted_instruction(_prog.instruction_at(*label)) ? 1ULL : 0ULL, .unbounded = false};
        }

        const auto& cycle = *std::get<std::shared_ptr<WtoCycle>>(component);
        if (!thread_local_options.cfg_opts.check_for_termination) {
            return unknown_unbounded();
        }

        uint64_t body_count = 0;
        for (const auto& sub_component : cycle) {
            const WorstInstrCount summary = summarize_component(sub_component);
            if (summary.unbounded) {
                return summary;
            }
            if (!checked_add(body_count, summary.count, body_count)) {
                return overflow_unbounded();
            }
        }

        const auto loop_bound = get_loop_bound(cycle.head());
        if (!loop_bound.has_value()) {
            return unknown_unbounded();
        }

        uint64_t loop_count = 0;
        if (!checked_mul(*loop_bound, body_count, loop_count)) {
            return overflow_unbounded();
        }
        return {.count = loop_count, .unbounded = false};
    }

    [[nodiscard]]
    WorstInstrCount worst_instr_count() const {
        const std::vector<CycleOrLabel> components{_wto.begin(), _wto.end()};
        std::vector<WorstInstrCount> summaries;
        summaries.reserve(components.size());
        std::map<Label, size_t> label_to_component;

        for (size_t i = 0; i < components.size(); ++i) {
            summaries.push_back(summarize_component(components[i]));
            std::vector<Label> labels;
            collect_labels(components[i], labels);
            for (const Label& label : labels) {
                label_to_component.emplace(label, i);
            }
        }

        std::vector<std::set<size_t>> successors(components.size());
        for (size_t i = 0; i < components.size(); ++i) {
            std::vector<Label> labels;
            collect_labels(components[i], labels);
            for (const Label& label : labels) {
                for (const Label& succ : _cfg.children_of(label)) {
                    const auto it = label_to_component.find(succ);
                    if (it != label_to_component.end() && it->second != i) {
                        successors[i].insert(it->second);
                    }
                }
            }
        }

        const auto entry_it = label_to_component.find(Label::entry);
        const auto exit_it = label_to_component.find(Label::exit);
        if (entry_it == label_to_component.end() || exit_it == label_to_component.end()) {
            return unknown_unbounded();
        }

        const size_t exit_index = exit_it->second;
        std::vector<PathSummary> path_summaries(components.size());
        for (size_t i = components.size(); i-- > 0;) {
            if (i == exit_index) {
                path_summaries[i] = {.reaches_exit = true, .count = 0, .unbounded = false};
                continue;
            }

            bool found_path = false;
            bool has_unbounded_path = summaries[i].unbounded;
            uint64_t unbounded_count = summaries[i].count == std::numeric_limits<uint64_t>::max()
                                           ? std::numeric_limits<uint64_t>::max()
                                           : 0;
            uint64_t best_bounded = 0;
            for (const size_t succ : successors[i]) {
                const PathSummary& succ_summary = path_summaries[succ];
                if (!succ_summary.reaches_exit) {
                    continue;
                }
                found_path = true;
                has_unbounded_path = has_unbounded_path || succ_summary.unbounded;
                if (succ_summary.unbounded) {
                    unbounded_count =
                        combine_unbounded(summaries[i], WorstInstrCount{.count = succ_summary.count, .unbounded = true})
                            .count;
                    continue;
                }
                if (!succ_summary.unbounded) {
                    uint64_t candidate = 0;
                    if (!checked_add(summaries[i].count, succ_summary.count, candidate)) {
                        has_unbounded_path = true;
                        unbounded_count = std::numeric_limits<uint64_t>::max();
                    } else {
                        best_bounded = std::max(best_bounded, candidate);
                    }
                }
            }

            if (!found_path) {
                continue;
            }

            if (has_unbounded_path) {
                path_summaries[i] = {.reaches_exit = true, .count = unbounded_count, .unbounded = true};
                continue;
            }

            path_summaries[i] = {.reaches_exit = true, .count = best_bounded, .unbounded = false};
        }

        const PathSummary& entry_summary = path_summaries[entry_it->second];
        if (!entry_summary.reaches_exit) {
            return unknown_unbounded();
        }
        return {.count = entry_summary.count, .unbounded = entry_summary.unbounded};
    }

  public:
    void operator()(const Label& node);

    void operator()(const std::shared_ptr<WtoCycle>& cycle);

    static AnalysisResult run(const Program& prog, EbpfDomain entry_inv);
};

AnalysisResult analyze(const Program& prog) {
    ebpf_verifier_clear_before_analysis();
    return InterleavedFwdFixpointIterator::run(prog, EbpfDomain::setup_entry(thread_local_options.setup_constraints));
}

AnalysisResult analyze(const Program& prog, const StringInvariant& entry_invariant) {
    ebpf_verifier_clear_before_analysis();
    return InterleavedFwdFixpointIterator::run(
        prog, EbpfDomain::from_constraints(entry_invariant.value(), thread_local_options.setup_constraints));
}

static EbpfDomain extrapolate(const EbpfDomain& before, const EbpfDomain& after, const unsigned int iteration) {
    /// number of iterations until triggering widening
    constexpr auto _widening_delay = 2;

    if (iteration < _widening_delay) {
        return before | after;
    }
    return before.widen(after, iteration == _widening_delay);
}

static EbpfDomain refine(const EbpfDomain& before, const EbpfDomain& after, const unsigned int iteration) {
    if (iteration == 1) {
        return before & after;
    } else {
        return before.narrow(after);
    }
}

void InterleavedFwdFixpointIterator::operator()(const Label& node) {
    /** decide whether skip vertex or not **/
    if (_skip && node == _cfg.entry_label()) {
        _skip = false;
    }
    if (_skip) {
        return;
    }

    EbpfDomain pre = join_all_prevs(node);

    set_pre(node, pre);
    transform_to_post(node, std::move(pre));
}

void InterleavedFwdFixpointIterator::operator()(const std::shared_ptr<WtoCycle>& cycle) {
    const Label head = cycle->head();

    /** decide whether to skip cycle or not **/
    bool entry_in_this_cycle = false;
    if (_skip) {
        // We only skip the analysis of cycle if entry_label is not a
        // component of it, included nested components.
        entry_in_this_cycle = is_component_member(_cfg.entry_label(), cycle);
        _skip = !entry_in_this_cycle;
        if (_skip) {
            return;
        }
    }

    EbpfDomain invariant = EbpfDomain::bottom();
    if (entry_in_this_cycle) {
        invariant = get_pre(_cfg.entry_label());
    } else {
        const WtoNesting cycle_nesting = _wto.nesting(head);
        for (const Label& prev : _cfg.parents_of(head)) {
            if (!(_wto.nesting(prev) > cycle_nesting)) {
                invariant |= get_post(prev);
            }
        }
    }

    for (unsigned int iteration = 1;; ++iteration) {
        // Increasing iteration sequence with widening
        set_pre(head, invariant);
        transform_to_post(head, invariant);
        for (const auto& component : *cycle) {
            const auto plabel = std::get_if<Label>(&component);
            if (!plabel || *plabel != head) {
                std::visit(*this, component);
            }
        }
        EbpfDomain new_pre = join_all_prevs(head);
        if (new_pre <= invariant) {
            // Post-fixpoint reached
            set_pre(head, new_pre);
            invariant = std::move(new_pre);
            break;
        } else {
            invariant = extrapolate(invariant, new_pre, iteration);
        }
    }

    for (unsigned int iteration = 1;; ++iteration) {
        // Decreasing iteration sequence with narrowing
        transform_to_post(head, invariant);

        for (const auto& component : *cycle) {
            const auto plabel = std::get_if<Label>(&component);
            if (!plabel || *plabel != head) {
                std::visit(*this, component);
            }
        }
        EbpfDomain new_pre = join_all_prevs(head);
        if (invariant <= new_pre) {
            // No more refinement possible(pre == new_pre)
            break;
        } else {
            if (iteration > _descending_iterations) {
                break;
            }
            invariant = refine(invariant, std::move(new_pre), iteration);
            set_pre(head, std::move(invariant));
        }
    }
}
AnalysisResult InterleavedFwdFixpointIterator::run(const Program& prog, EbpfDomain entry_inv) {
    // Go over the CFG in weak topological order (accounting for loops).
    AnalysisResult result;
    InterleavedFwdFixpointIterator analyzer(prog, result);
    if (thread_local_options.cfg_opts.check_for_termination) {
        // Initialize loop counters for potential loop headers.
        // This enables enforcement of upper bounds on loop iterations
        // during program verification.
        // TODO: Consider making this an instruction instead of an explicit call.
        analyzer._wto.for_each_loop_head(
            [&](const Label& label) { ebpf_domain_initialize_loop_counter(entry_inv, label); });
    }
    analyzer.set_pre(prog.cfg().entry_label(), std::move(entry_inv));
    for (const auto& component : analyzer._wto) {
        std::visit(analyzer, component);
    }
    const WorstInstrCount worst_instr_count = analyzer.worst_instr_count();
    result.worst_instr_count = worst_instr_count.count;
    result.worst_instr_count_unbounded = worst_instr_count.unbounded ? 1 : 0;
    if (!result.failed && thread_local_options.cfg_opts.check_for_termination) {
        analyzer.find_termination_errors(prog);
        if (!result.failed) {
            result.max_loop_count = analyzer.max_loop_count();
        }
    }
    result.exit_value = analyzer.get_post(Label::exit).get_r0();
    return result;
}

} // namespace prevail
