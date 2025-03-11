using DrWatson
@quickactivate "DiffEqGMRFs"

using DataFrames, TimerOutputs, Distributions, CairoMakie, TuePlots, LaTeXStrings
CairoMakie.activate!()

df = collect_results(datadir("sims", "elliptic-chen"))
df_linear = filter(row -> row.el_order == 1, df)
sort!(df_linear, [:N_el_xy])
df_quadratic = filter(row -> row.el_order == 2, df)
sort!(df_quadratic, [:N_el_xy])

function get_times_and_errs(df)
    N_el_xys = df[!, "N_el_xy"]
    tos = df[!, "to"]
    times = [TimerOutputs.time(to["Solve time"]) / 1e9 for to in tos]
    errs_L2 = df[!, "err_L2"]
    errs_MAE = df[!, "err_MAE"]
    errs_rel = df[!, "err_rel"]
    return N_el_xys, times, errs_L2, errs_MAE, errs_rel
end

N_el_xys_linear, times_linear, errs_L2_linear, errs_MAE_linear, errs_rel_linear = get_times_and_errs(df_linear)
N_el_xys_quadratic, times_quadratic, errs_L2_quadratic, errs_MAE_quadratic, errs_rel_quadratic = get_times_and_errs(df_quadratic)

using PyCall
using PyPlot
bundles = pyimport("tueplots.bundles")
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
merge!(rcParams, bundles.aistats2025(column="full", nrows=1, ncols=2))

plt = pyimport("matplotlib.pyplot")
bundles = pyimport("tueplots.bundles")
merge!(plt.rcParams, bundles.aistats2025(column="full", nrows=1, ncols=2))

println(N_el_xys_linear)
println(times_linear)
println(N_el_xys_quadratic)
println(times_quadratic)

fig, axes = subplots(1, 2)
ax_L2 = axes[1]
ax_time = axes[2]

ax_L2.set_yscale("log")
ax_L2.set_xticks(N_el_xys_linear)
ax_L2.set_xlabel(L"N_{xy}")
ax_L2.set_ylabel("L2 error")

ax_time.set_yscale("log")
ax_time.set_xticks(N_el_xys_linear)
ax_time.set_xlabel(L"N_{xy}")
ax_time.set_yticks([1, 5, 10, 20])
ax_time.set_ylabel("CPU time (s)")

ax_L2.plot(N_el_xys_linear, errs_L2_linear, color="red", label="Linear", marker="o")
ax_L2.plot(N_el_xys_quadratic, errs_L2_quadratic, color="blue", marker="s", label="Quadratic")

ax_time.plot(N_el_xys_linear, times_linear, color="red", label="Linear", marker="o")
ax_time.plot(N_el_xys_quadratic, times_quadratic, color="blue", marker="s", label="Quadratic")

ax_L2.grid()
ax_time.grid()
ax_L2.legend(loc="center right")
ax_time.legend(loc="lower right")

save_path = projectdir("plots", "chen_elliptic_results.pdf")
savefig(save_path)

#with_theme(T) do
    ##size_inches = (10.4, 4)
    ##size_pt = 72 .* size_inches
#
    ##fig = Figure(resolution=size_pt, fontsize=12)
    #fig = Figure()
    #ax_L2 = Axis(fig[1, 1], yscale=log10, xticks=N_el_xys_linear, xlabel=L"N_{xy}", ylabel="L2 error")
    #ax_time = Axis(fig[1, 2], yscale=log10, xticks=N_el_xys_linear, xlabel=L"N_{xy}", yticks=[2, 5, 10, 20], ylabel="CPU time (s)")
#
    #scatterlines!(ax_L2, N_el_xys_linear, errs_L2_linear, color=:red, label="Linear")
    #scatterlines!(ax_L2, N_el_xys_quadratic, errs_L2_quadratic, color=:blue, marker=:rect, label="Quadratic")
#
    #scatterlines!(ax_time, N_el_xys_linear, times_linear, color=:red, label="Linear")
    #scatterlines!(ax_time, N_el_xys_quadratic, times_quadratic, color=:blue, marker=:rect, label="Quadratic")
#
    #axislegend(ax_L2, position=:rc)
    #axislegend(ax_time, position=:rb)
#
    #save_path = projectdir("plots", "chen_elliptic_results.pdf")
    #save(save_path, fig, pt_per_unit = 1)
#end
#
