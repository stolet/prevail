// Copyright (c) Prevail Verifier contributors.
// SPDX-License-Identifier: MIT
#include <catch2/catch_all.hpp>

#include <gsl/narrow>

#include "ir/parse.hpp"
#include "ir/program.hpp"
#include "verifier.hpp"

using namespace prevail;

namespace {
using RawBlocks = std::vector<std::tuple<std::string, std::vector<std::string>>>;

InstructionSeq raw_cfg_to_instruction_seq(const RawBlocks& raw_blocks) {
    std::map<std::string, Label> label_name_to_label;

    int label_index = 0;
    for (const auto& [label_name, raw_block] : raw_blocks) {
        label_name_to_label.emplace(label_name, label_index);
        label_index += gsl::narrow<int>(raw_block.size());
    }

    InstructionSeq res;
    label_index = 0;
    for (const auto& [_, raw_block] : raw_blocks) {
        for (const std::string& line : raw_block) {
            res.emplace_back(label_index++, parse_instruction(line, label_name_to_label), std::optional<btf_line_info_t>{});
        }
    }
    return res;
}

Program make_program(const RawBlocks& raw_blocks, const ebpf_verifier_options_t& options) {
    const ProgramInfo info{};
    return Program::from_sequence(raw_cfg_to_instruction_seq(raw_blocks), info, options);
}

AnalysisResult analyze_program(const RawBlocks& raw_blocks, const ebpf_verifier_options_t& options,
                               const std::set<std::string>& pre = {}) {
    ThreadLocalGuard guard;
    const Program prog = make_program(raw_blocks, options);
    if (pre.empty()) {
        return analyze(prog);
    }
    return analyze(prog, StringInvariant{pre});
}
} // namespace

TEST_CASE("worst instruction count for straight-line program", "[analysis][worst_instr_count]") {
    ebpf_verifier_options_t options{};
    options.setup_constraints = false;

    const AnalysisResult result = analyze_program(
        {
            {"<start>", {"r0 = 0", "r0 += 1"}},
            {"<out>", {"exit"}},
        },
        options);

    REQUIRE(result.worst_instr_count == 3);
    REQUIRE(result.worst_instr_count_unbounded == 0);
}

TEST_CASE("worst instruction count chooses longer branch", "[analysis][worst_instr_count]") {
    ebpf_verifier_options_t options{};
    options.setup_constraints = false;

    const AnalysisResult result = analyze_program(
        {
            {"<start>", {"if r1 == 0 goto <out>"}},
            {"<long>", {"r0 = 0", "r0 += 1"}},
            {"<out>", {"exit"}},
        },
        options, {"r1.type=number"});

    REQUIRE(result.worst_instr_count == 4);
    REQUIRE(result.worst_instr_count_unbounded == 0);
}

TEST_CASE("worst instruction count over-approximates bounded loop bodies", "[analysis][worst_instr_count]") {
    ebpf_verifier_options_t options{};
    options.cfg_opts.check_for_termination = true;
    options.setup_constraints = false;

    const AnalysisResult result = analyze_program(
        {
            {"<start>", {"r0 = 0"}},
            {"<loop>", {"if r0 >= 4 goto <out>", "r0 += 1", "goto <loop>"}},
            {"<out>", {"exit"}},
        },
        options);

    REQUIRE_FALSE(result.failed);
    REQUIRE(result.worst_instr_count == 17);
    REQUIRE(result.worst_instr_count_unbounded == 0);
}

TEST_CASE("worst instruction count is unbounded for cyclic programs without termination checking",
          "[analysis][worst_instr_count]") {
    ebpf_verifier_options_t options{};
    options.setup_constraints = false;

    const AnalysisResult result = analyze_program(
        {
            {"<start>", {"r0 = 0", "if r0 < 1 goto <start>"}},
            {"<out>", {"exit"}},
        },
        options);

    REQUIRE(result.worst_instr_count == 0);
    REQUIRE(result.worst_instr_count_unbounded == 1);
}
