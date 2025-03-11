using DrWatson
@quickactivate "DiffEqGMRFs"

using DataFrames, TimerOutputs, Distributions
df = collect_results(datadir("sims", "burgers", "gmrf-collocation"))
df = filter(row -> row.dry_run == false, df)
df_adv_diff = filter(row -> row.prior_type == "adv_diff", df)
df_product_matern = filter(row -> row.prior_type == "product_matern", df)
sort!(df_adv_diff, [:N_collocation])
sort!(df_product_matern, [:N_collocation])

function get_errs(df)
    errs_rel = df[!, :rel_errs]
    return errs_rel
end

function get_ic_errs(df)
    ic_rel_errs = df[!, :ic_rel_errs]
    return ic_rel_errs
end

errs_rel_adv_diff = get_errs(df_adv_diff)
errs_rel_product_matern = get_errs(df_product_matern)

num_col = [0, 5, 10, 25, 100, 250, 500, 1000]
ic_rel_errs_ad = mean.(get_ic_errs(df_adv_diff))
ic_rel_errs_pm = mean.(get_ic_errs(df_product_matern))
ic_rel_errs_ad_std = std.(get_ic_errs(df_adv_diff))
ic_rel_errs_pm_std = std.(get_ic_errs(df_product_matern))
rel_errs_ad = mean.(errs_rel_adv_diff)
rel_errs_pm = mean.(errs_rel_product_matern)
std_rel_errs_ad = std.(errs_rel_adv_diff)
std_rel_errs_pm = std.(errs_rel_product_matern)

errs_ad = [ic_rel_errs_ad[1]] ∪ rel_errs_ad
errs_pm = [ic_rel_errs_pm[1]] ∪ rel_errs_pm
stds_ad = [ic_rel_errs_ad_std[1]] ∪ std_rel_errs_ad
stds_pm = [ic_rel_errs_pm_std[1]] ∪ std_rel_errs_pm

# Print error in % +- std
for i in 1:length(num_col)
    println(num_col[i])
    println("Advection-Diffusion: $(round(errs_ad[i] * 100, digits=2)) ± $(round(stds_ad[i] * 100, digits=2))")
    println("Product Matérn: $(round(errs_pm[i] * 100, digits=2)) ± $(round(stds_pm[i] * 100, digits=2))")
end


#println(errs_ad)
#println(errs_pm)
#println(stds_ad)
#println(stds_pm)

errs_ad = errs_ad .* 100
errs_pm = errs_pm .* 100

using PyCall
using PyPlot
bundles = pyimport("tueplots.bundles")
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
merge!(rcParams, bundles.aistats2025(column="half"))

figure()
plot(num_col, errs_ad, "o-", label="Advection-Diffusion")
plot(num_col, errs_pm, "s-", label="Product Matérn")
xlabel("Number of collocation points")
ylabel(raw"Relative error (\%)")
title("1D Burgers' equation")
legend()
grid()

save_path = projectdir("plots", "burgers_collocation_plot.pdf")
savefig(save_path)
