// KernRelay 外部库入口：把现有 CLI 入口收敛为稳定 C ABI。
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int RunPlanCli(int argc, char **argv);

extern "C" const char *kr_version(void)
{
    return "KernRelay 0.3 (libkernrelay)";
}

static void SetEnv(const char *key, const char *value, bool on)
{
    if (on) {
        setenv(key, value, 1);
    } else {
        unsetenv(key);
    }
}

extern "C" int kr_plan_run(const char *schedule_json, int iters, int warmup,
                           const char *devices, const char *mode,
                           const char *out_json)
{
    if (!schedule_json || !out_json) return 2;
    if (devices && *devices) SetEnv("NHP_DEVICES", devices, true);
    else SetEnv("NHP_DEVICES", "", false);

    const std::string m = mode ? mode : "direct";
    SetEnv("NHP_GRAPH", "1", m == "graph");
    SetEnv("NHP_GRAPH_PER_STREAM", "1", m == "graph_per_stream");
    SetEnv("NHP_GRAPH_SEGMENT", "1", m == "graph_segment");

    std::vector<std::string> args;
    args.push_back("libkernrelay");
    args.push_back(schedule_json);
    args.push_back(std::to_string(iters));
    args.push_back(std::to_string(warmup));
    args.push_back(out_json);
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (auto &s : args) argv.push_back(const_cast<char *>(s.c_str()));
    return RunPlanCli(static_cast<int>(argv.size()), argv.data());
}
