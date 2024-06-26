#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
# SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

declare -A run_flags
    run_flags[problem]=euler_vortex
    run_flags[degree]=2
    run_flags[dm_plex_box_faces]=20,20,1
    run_flags[lx]=1e3
    run_flags[ly]=1e3
    run_flags[lz]=1
    run_flags[ts_max_time]=.02
    run_flags[ts_rk_type]=5bs
    run_flags[ts_rtol]=1e-10
    run_flags[ts_atol]=1e-10

declare -A test_flags
    test_flags[degree_start]=1
    test_flags[degree_stride]=1
    test_flags[degree_end]=2
    test_flags[res_start]=6
    test_flags[res_stride]=2
    test_flags[res_end]=10

file_name=conv_test_result.csv

echo ",mesh_res,degree,rel_error" > $file_name

i=0
for ((d=${test_flags[degree_start]}; d<=${test_flags[degree_end]}; d+=${test_flags[degree_stride]})); do
    run_flags[degree]=$d
    for ((res=${test_flags[res_start]}; res<=${test_flags[res_end]}; res+=${test_flags[res_stride]})); do
        run_flags[dm_plex_box_faces]=$res,$res,1
        args=''
        for arg in "${!run_flags[@]}"; do
            if ! [[ -z ${run_flags[$arg]} ]]; then
                args="$args -$arg ${run_flags[$arg]}"
            fi
        done
        ./navierstokes $args | grep "Relative Error:" | awk -v i="$i" -v res="$res" -v d="$d" '{ printf "%d,%d,%d,%.5f\n", i, res, d, $3}' >> $file_name
        i=$((i+1))
    done
done

# Compare the output CSV file with the reference file
count=$(diff conv_test_result.csv tests-output/fluids-navierstokes-conv-euler.csv | grep "^>" | wc -l)
if [ ${count} != 0 ]; then
    printf "\n# TEST FAILED!\n\n"
    diff -q conv_test_result.csv tests-output/fluids-navierstokes-conv-euler.csv
    printf "\n"
fi
