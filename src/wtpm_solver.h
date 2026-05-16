//   ______  _______   _______   ______  __
//  /      ||       \ |   ____| /      ||  |
// |  ,----'|  .--.  ||  |__   |  ,----'|  |
// |  |     |  |  |  ||   __|  |  |     |  |
// |  `----.|  '--'  ||  |     |  `----.|  |
//  \______||_______/ |__|      \______||__|
//
// WTPM-CD solver — optimized with template dispatch and merged lookups
// Based on: Gao, Li, Shen, SIAM J. Sci. Comput., 46(1), A179-A203, 2024

#ifndef CDFCI_WTPM_SOLVER_H
#define CDFCI_WTPM_SOLVER_H 1

#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include "wtpm_wavefunction.h"
#include "hamiltonian.h"
#include "hash.h"

inline double solve_cubic_wtpm(double c1, double c0)
{
    const double pi = atan(1.0) * 4;
    double p3 = c1 / 3.0;
    double q2 = c0 / 2.0;
    double disc = p3 * p3 * p3 + q2 * q2;
    double t = 0;

    if (disc >= 0)
    {
        double qrtd = sqrt(disc);
        t = cbrt(-q2 + qrtd) + cbrt(-q2 - qrtd);
    }
    else
    {
        double qrtd = sqrt(-disc);
        if (q2 >= 0)
            t = 2.0 * sqrt(-p3) * cos((atan2(-qrtd, -q2) - 2.0 * pi) / 3.0);
        else
            t = 2.0 * sqrt(-p3) * cos(atan2(qrtd, -q2) / 3.0);
    }

    if (fabs(t) > 1e-300)
    {
        double tn = t - (t * (t * t + c1) + c0) / (3.0 * t * t + c1);
        int iter = 0;
        while (fabs((tn - t) / fabs(t)) > 1e-12 && iter < 10)
        {
            t = tn;
            tn = t - (t * (t * t + c1) + c0) / (3.0 * t * t + c1);
            ++iter;
        }
        t = tn;
    }
    return t;
}

template<int N = 1>
struct DiagDetCompare {
    bool operator()(const std::pair<double, Determinant<N>>& a,
                    const std::pair<double, Determinant<N>>& b) const {
        return a.first < b.first;
    }
};

template<int N = 1>
class WTPMCD
{
    public:

    Option option;
    int p;
    double mu;
    int num_iter;
    int report_interval;
    double z_threshold;
    bool z_threshold_search;
    size_t max_wf_size;

    std::vector<double> result_energies;
    Determinant<N> last_picked_det;

    WTPMCD(Option& opt)
    {
        p = opt.value("p", 1);
        mu = opt.value("mu", 1.0);
        num_iter = opt.value("num_iterations", 30000);
        report_interval = opt.value("report_interval", 1000);
        z_threshold = opt.value("z_threshold", 0.0);
        z_threshold_search = opt.value("z_threshold_search", false);
        max_wf_size = opt.value("max_wavefunction_size", 1000000);
        result_energies.resize(p, 0.0);
        option = opt;
    }

    ~WTPMCD() {}

    double get_energy(int l) const { return result_energies[l]; }

    void init_weights(WTPMWaveFunctionBase<N>& wf, const std::vector<double>& rayleigh_quotients)
    {
        if (static_cast<int>(rayleigh_quotients.size()) < p)
            throw std::invalid_argument("Not enough Rayleigh quotients to initialize weights.");

        auto r = rayleigh_quotients;
        std::sort(r.begin(), r.begin() + p);

        double epsilon = fabs(r[p - 1] - r[0]) * 0.01;
        if (epsilon < 1e-10) epsilon = 1e-6;
        double wp = r[p - 1] + epsilon;
        double w1 = 2.0 * wp - r[0];

        if (p == 1)
        {
            wf.W[0] = wp;
        }
        else
        {
            for (int i = 0; i < p; ++i)
                wf.W[i] = w1 - (w1 - wp) * static_cast<double>(i) / static_cast<double>(p - 1);
        }
    }

    std::vector<Determinant<N>> find_initial_dets(Hamiltonian<N>& h)
    {
        auto hf = h.get_hartree_fock();
        DeterminantDecoded<N> hf_decoded(hf);

        std::vector<std::pair<double, Determinant<N>>> diag_dets;

        double hf_diag = h.get_diagonal(hf_decoded);
        diag_dets.push_back({hf_diag, hf});

        auto column = h.get_column(hf_decoded);
        for (auto& entry : column)
        {
            DeterminantDecoded<N> dec(entry.first);
            double diag = h.get_diagonal(dec);
            diag_dets.push_back({diag, entry.first});
        }

        DiagDetCompare<N> cmp;
        std::stable_sort(diag_dets.begin(), diag_dets.end(), cmp);

        std::vector<Determinant<N>> result;
        for (auto& item : diag_dets)
        {
            bool is_new = true;
            for (auto& existing : result)
            {
                if (DeterminantEqual<N>()(item.second, existing))
                {
                    is_new = false;
                    break;
                }
            }
            if (is_new)
            {
                result.push_back(item.second);
                if (static_cast<int>(result.size()) >= p) break;
            }
        }

        while (static_cast<int>(result.size()) < p)
            result.push_back(hf);

        return result;
    }

    template<typename WFType>
    int solve_impl(Hamiltonian<N>& h, WFType& wf)
    {
        auto init_dets = find_initial_dets(h);

        std::vector<double> init_rq(p);
        for (int l = 0; l < p; ++l)
        {
            DeterminantDecoded<N> dec(init_dets[l]);
            init_rq[l] = h.get_diagonal(dec);
        }
        init_weights(wf, init_rq);

        std::cout << "Weight matrix W: ";
        for (int l = 0; l < p; ++l)
            std::cout << std::fixed << std::setprecision(3) << wf.W[l] << " ";
        std::cout << std::endl << std::endl;

        alignas(64) double x_k_all[WTPM_MAX_P];
        alignas(64) double x_all_buf[WTPM_MAX_P];

        // Initialize X and Y
        for (int l = 0; l < p; ++l)
        {
            DeterminantDecoded<N> dec(init_dets[l]);
            double diag = h.get_diagonal(dec);

            wf.get_x(init_dets[l], x_k_all);
            double dx = 1.0 - x_k_all[l];
            wf.update_x(init_dets[l], l, dx, x_k_all);

            auto column = h.get_column(dec);
            WTPMWaveFunctionVector<N> sub_xz;
            wf.update_z_and_get_sub(column, 1.0, l, diag, sub_xz, z_threshold);

            wf.d[l] = diag;

            double exact_z = 0;
            for (auto& entry : column)
            {
                double xv, zv;
                if (wf.get_xz(entry.first, l, xv, zv))
                    exact_z += entry.second * xv;
            }
            wf.reinsert_z(init_dets[l], l, exact_z);
        }

        // Initialize S
        for (int i = 0; i < p; ++i)
            for (int j = i; j < p; ++j)
            {
                if (DeterminantEqual<N>()(init_dets[i], init_dets[j]))
                {
                    wf.S_ref(i, j) = 1.0;
                    wf.S_ref(j, i) = 1.0;
                }
                else
                {
                    wf.S_ref(i, j) = 0.0;
                    wf.S_ref(j, i) = 0.0;
                }
            }

        last_picked_det = init_dets[0];

        std::cout << "Initial energies:" << std::endl;
        for (int l = 0; l < p; ++l)
            std::cout << "  State " << l << ": " << std::fixed << std::setprecision(10)
                      << wf.get_energy(l) << std::endl;
        std::cout << std::endl;

        std::cout << std::setw(13) << std::left << "Iteration";
        for (int l = 0; l < p; ++l)
            std::cout << std::setw(18) << std::right << ("E_" + std::to_string(l));
        std::cout << std::setw(18) << "dx";
        std::cout << std::setw(15) << "|Y|_0";
        std::cout << std::setw(10) << "Time";
        std::cout << std::endl;

        auto time_start = std::chrono::high_resolution_clock::now();
        double dx = 0;

        int n_outer = num_iter / report_interval;

#ifdef _OPENMP
        for (int ri = 0; ri < n_outer; ++ri)
        {
            for (int j = 0; j < report_interval; ++j)
            {
                int iter = ri * report_interval + j;
                int l = iter % p;

                DeterminantDecoded<N> last_decoded(last_picked_det);
                auto search_column = h.get_column(last_decoded);

                double max_grad = 0;
                Determinant<N> picked_det;
                double picked_x = 0, picked_z = 0;

                for (auto& entry : search_column)
                {
                    auto& det = entry.first;
                    double x_val, z_val;
                    wf.get_x_and_xz(det, l, x_all_buf, x_val, z_val);

                    double grad = z_val;
                    for (int s = 0; s < p; ++s)
                        grad += mu * x_all_buf[s] * (wf.S_ref(s, l) - (s == l ? wf.W[l] : 0.0));

                    double abs_grad = fabs(grad);
                    if (abs_grad >= max_grad)
                    {
                        max_grad = abs_grad;
                        picked_det = det;
                        picked_x = x_val;
                        picked_z = z_val;
                        for (int s = 0; s < p; ++s) x_k_all[s] = x_all_buf[s];
                    }
                }

                if (max_grad == 0) continue;

                last_picked_det = picked_det;

                DeterminantDecoded<N> picked_decoded(picked_det);
                double A_kk = h.get_diagonal(picked_decoded);

                double sum_xml_sq = wf.S_ref(l, l);
                double sum_xks_sq = 0;
                for (int s = 0; s < p; ++s)
                    sum_xks_sq += x_k_all[s] * x_k_all[s];

                double x_kl = picked_x;

                double c1 = (1.0 / mu) * (A_kk - wf.W[l] + sum_xml_sq + sum_xks_sq - 2.0 * x_kl * x_kl);

                double sum_sxSl = 0;
                for (int s = 0; s < p; ++s)
                    sum_sxSl += x_k_all[s] * wf.S_ref(s, l);

                double c0 = (1.0 / mu) * (picked_z - A_kk * x_kl)
                          + sum_sxSl
                          + x_kl * x_kl * x_kl
                          - x_kl * (sum_xks_sq + sum_xml_sq);

                double t = solve_cubic_wtpm(c1, c0);
                dx = t - x_kl;

                wf.update_x(picked_det, l, dx, x_k_all);

                auto h_column_ds = h.get_column_diagonal_single(picked_decoded);
                WTPMWaveFunctionVector<N> sub_xz;
                wf.update_z_and_get_sub(h_column_ds, dx, l, A_kk, sub_xz, z_threshold);
                double Y_new_kl = sub_xz.new_z;

                #pragma omp parallel shared(picked_decoded, dx, h, wf, Y_new_kl, l)
                {
                    typename Hamiltonian<N>::Column column_double;
                    WTPMWaveFunctionVector<N> sub_xz_double;

                    #pragma omp for schedule(dynamic, 4)
                    for (auto idx = 0; idx < h.nelec * (h.nelec - 1) / 2; ++idx)
                    {
                        int idx_j = 0.5 + sqrt(0.25 + 2 * idx);
                        int idx_i = idx - (idx_j - 1) * idx_j / 2;
                        h.get_double_excitation(picked_decoded, column_double, idx_i, idx_j);
                    }

                    wf.update_z_only(column_double, dx, l, sub_xz_double, z_threshold);

                    #pragma omp atomic update
                    Y_new_kl += sub_xz_double.new_z;
                }

                wf.update_d(l, dx, Y_new_kl, A_kk);
                wf.reinsert_z(picked_det, l, Y_new_kl);
            }

            auto time_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = time_end - time_start;

            std::cout << std::setw(13) << std::left << (ri + 1) * report_interval;
            for (int l = 0; l < p; ++l)
            {
                double e = wf.get_energy(l);
                result_energies[l] = e;
                std::cout << std::setw(18) << std::right << std::fixed << std::setprecision(10) << e;
            }
            std::cout << std::setw(18) << std::scientific << std::setprecision(4) << dx;
            std::cout << std::setw(15) << wf.size();
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << elapsed.count();
            std::cout << std::endl;

            if (wf.size() > 0.79 * wf.max_size)
                throw std::overflow_error("The hash table is full. Please increase max_memory or z_threshold.");
        }
#else
        for (int ri = 0; ri < n_outer; ++ri)
        {
            for (int j = 0; j < report_interval; ++j)
            {
                int iter = ri * report_interval + j;
                int l = iter % p;

                DeterminantDecoded<N> last_decoded(last_picked_det);
                auto search_column = h.get_column(last_decoded);

                double max_grad = 0;
                Determinant<N> picked_det;
                double picked_x = 0, picked_z = 0;

                for (auto& entry : search_column)
                {
                    auto& det = entry.first;
                    double x_val, z_val;
                    wf.get_x_and_xz(det, l, x_all_buf, x_val, z_val);

                    double grad = z_val;
                    for (int s = 0; s < p; ++s)
                        grad += mu * x_all_buf[s] * (wf.S_ref(s, l) - (s == l ? wf.W[l] : 0.0));

                    double abs_grad = fabs(grad);
                    if (abs_grad >= max_grad)
                    {
                        max_grad = abs_grad;
                        picked_det = det;
                        picked_x = x_val;
                        picked_z = z_val;
                        for (int s = 0; s < p; ++s) x_k_all[s] = x_all_buf[s];
                    }
                }

                if (max_grad == 0) continue;

                last_picked_det = picked_det;

                DeterminantDecoded<N> picked_decoded(picked_det);
                double A_kk = h.get_diagonal(picked_decoded);

                double sum_xml_sq = wf.S_ref(l, l);
                double sum_xks_sq = 0;
                for (int s = 0; s < p; ++s)
                    sum_xks_sq += x_k_all[s] * x_k_all[s];

                double x_kl = picked_x;

                double c1 = (1.0 / mu) * (A_kk - wf.W[l] + sum_xml_sq + sum_xks_sq - 2.0 * x_kl * x_kl);

                double sum_sxSl = 0;
                for (int s = 0; s < p; ++s)
                    sum_sxSl += x_k_all[s] * wf.S_ref(s, l);

                double c0 = (1.0 / mu) * (picked_z - A_kk * x_kl)
                          + sum_sxSl
                          + x_kl * x_kl * x_kl
                          - x_kl * (sum_xks_sq + sum_xml_sq);

                double t = solve_cubic_wtpm(c1, c0);
                dx = t - x_kl;

                wf.update_x(picked_det, l, dx, x_k_all);

                auto h_column = h.get_column(picked_decoded);
                WTPMWaveFunctionVector<N> sub_xz;
                wf.update_z_and_get_sub(h_column, dx, l, A_kk, sub_xz, z_threshold);

                wf.update_d(l, dx, sub_xz.new_z, A_kk);
                wf.reinsert_z(picked_det, l, sub_xz.new_z);
            }

            auto time_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = time_end - time_start;

            std::cout << std::setw(13) << std::left << (ri + 1) * report_interval;
            for (int l = 0; l < p; ++l)
            {
                double e = wf.get_energy(l);
                result_energies[l] = e;
                std::cout << std::setw(18) << std::right << std::fixed << std::setprecision(10) << e;
            }
            std::cout << std::setw(18) << std::scientific << std::setprecision(4) << dx;
            std::cout << std::setw(15) << wf.size();
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << elapsed.count();
            std::cout << std::endl;

            if (wf.size() > 0.79 * wf.max_size)
                throw std::overflow_error("The hash table is full. Please increase max_memory or z_threshold.");
        }
#endif

        for (int l = 0; l < p; ++l)
            result_energies[l] = wf.get_energy(l);

        return 0;
    }

    int solve(Hamiltonian<N>& h)
    {
        std::cout << "WTPM-CD calculation" << std::endl;
        std::cout << "-------------------" << std::endl;
        std::cout << "Number of eigenpairs: " << p << std::endl;
        std::cout << "Penalty parameter mu: " << mu << std::endl;
        std::cout << std::endl;

#ifdef _OPENMP
        WTPMWaveFunctionCuckoo<N> wf(max_wf_size, p, mu);
#else
        WTPMWaveFunctionStd<N> wf(max_wf_size, p, mu);
#endif
        return solve_impl(h, wf);
    }
};

#endif
