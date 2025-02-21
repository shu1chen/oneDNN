#include <assert.h>

#include <chrono>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

#ifdef _WIN32
#define PSAPI_VERSION 1 // PrintMemoryInfo 
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#include <psapi.h>
#pragma comment(lib,"psapi.lib") //PrintMemoryInfo
#include <stdio.h>
#include "processthreadsapi.h"

// To ensure correct resolution of symbols, add Psapi.lib to TARGETLIBS
// and compile with -DPSAPI_VERSION=1
static double GetMemoryInfo() {
    PROCESS_MEMORY_COUNTERS_EX2 pmc;
    double private_working_set_size_mb;
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc))) {
        private_working_set_size_mb = pmc.PrivateWorkingSetSize / 1024 / 1024.0;
    }

    return private_working_set_size_mb;
}

#else

#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
static double GetMemoryInfo() {
    std::ifstream status_file("/proc/self/status");
    std::string line;
    size_t vm_rss_kb = 0;

    if (status_file.is_open()) {
        while (std::getline(status_file, line)) {
            if (line.find("VmRSS:") == 0) {
                sscanf(line.c_str(), "VmRSS: %zu kB", &vm_rss_kb);
            }
        }
        status_file.close();
    }
    return vm_rss_kb / 1024.0;
}
#endif

using namespace dnnl;

void simple_net(std::string engine_kind_str, int times = 100) {
    engine::kind engine_kind;
    if (engine_kind_str == "cpu") {
        engine_kind = dnnl::engine::kind::cpu;
    } else if (engine_kind_str == "gpu") {
        engine_kind = dnnl::engine::kind::gpu;
    }

    using tag = memory::format_tag;
    using dt = memory::data_type;

    engine eng(engine_kind, 0);
    stream s(eng);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;


    const memory::dim batch = 1;

    // fc6 inner product {batch, 256, 6, 6} (x) {4096, 256, 6, 6}-> {batch,
    // 4096}
    memory::dims fc6_src_tz = {batch, 256, 6, 6};
    memory::dims fc6_weights_tz = {4096, 256, 6, 6};
    memory::dims fc6_bias_tz = {4096};
    memory::dims fc6_dst_tz = {batch, 4096};

    std::vector<float> fc6_weights(product(fc6_weights_tz));
    std::vector<float> fc6_bias(product(fc6_bias_tz));

    // create memory for user data
    auto fc6_user_weights_memory
            = memory({{fc6_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(fc6_weights.data(), fc6_user_weights_memory);
    auto fc6_user_bias_memory = memory({{fc6_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(fc6_bias.data(), fc6_user_bias_memory);

    // create memory descriptors for convolution data w/ no specified format
    auto fc6_src_md = memory::desc({fc6_src_tz}, dt::f32, tag::any);
    auto fc6_bias_md = memory::desc({fc6_bias_tz}, dt::f32, tag::any);
    auto fc6_weights_md = memory::desc({fc6_weights_tz}, dt::f32, tag::any);
    auto fc6_dst_md = memory::desc({fc6_dst_tz}, dt::f32, tag::any);

    // create a inner_product
    auto fc6_prim_desc = inner_product_forward::primitive_desc(eng,
            prop_kind::forward_inference, fc6_src_md, fc6_weights_md,
            fc6_bias_md, fc6_dst_md);

    auto fc6_src_memory = memory(fc6_prim_desc.src_desc(), eng);

    auto fc6_weights_memory = fc6_user_weights_memory;
    if (fc6_prim_desc.weights_desc() != fc6_user_weights_memory.get_desc()) {
        fc6_weights_memory = memory(fc6_prim_desc.weights_desc(), eng);
        reorder(fc6_user_weights_memory, fc6_weights_memory)
                .execute(s, fc6_user_weights_memory, fc6_weights_memory);
    }

    auto fc6_dst_memory = memory(fc6_prim_desc.dst_desc(), eng);

    // create convolution primitive and add it to net
    net.push_back(inner_product_forward(fc6_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, fc6_src_memory},
            {DNNL_ARG_WEIGHTS, fc6_weights_memory},
            {DNNL_ARG_BIAS, fc6_user_bias_memory},
            {DNNL_ARG_DST, fc6_dst_memory}});

    for (int j = 0; j < times; ++j) {
        assert(net.size() == net_args.size() && "something is missing");
        for (size_t i = 0; i < net.size(); ++i)
            net.at(i).execute(s, net_args.at(i));
    }
    //[Execute model]

    s.wait();
}

void cnn_inference_f32(std::string engine_kind_str) {
    auto begin = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    int times = 100;
    simple_net(engine_kind_str, times);
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
                       .count();
    std::cout << "Use time: " << (end - begin) / (times + 0.0)
              << " ms per iteration." << std::endl;
}

void check_memory_size(std::string engine_kind_str, int rounds = 10) {
    dnnl::set_primitive_cache_capacity(0);

    std::cout << "No workload, ";
    auto private_working_set_size_mb = GetMemoryInfo();
    std::cout << "PrivateWorkingSet Memory size: "
              << private_working_set_size_mb << " (MB)\n\n";

	// std::ofstream csv_file("memory_usage.csv");
    // csv_file << "Round,PrivateWorkingSet Memory size (MB)\n";

	int times = 100;
	for (int i = 0; i < rounds; i++) {
		std::cout << "Round: " << i << ", ";
		simple_net(engine_kind_str, times);
#ifdef _WIN32
		Sleep(1000);
#else
		sleep(1);
#endif
		private_working_set_size_mb = GetMemoryInfo();
		std::cout << "PrivateWorkingSet Memory size: "
				<< private_working_set_size_mb << " (MB)\n";

        // Write to CSV file
        // csv_file << i << "," << private_working_set_size_mb << "\n";
	}
}

int main(int argc, char **argv) {
    std::string engine_kind_str;
    if (argc == 1) {
        engine_kind_str = "cpu";
    } else {
        engine_kind_str = argv[1];
    }
	
	int rounds = 10;
	if (argc == 3) {
		rounds = atoi(argv[2]);
	}
    check_memory_size(engine_kind_str, rounds);

    return 0;
}
