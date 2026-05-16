//   ______  _______   _______   ______  __
//  /      ||       \ |   ____| /      ||  |
// |  ,----'|  .--.  ||  |__   |  ,----'|  |
// |  |     |  |  |  ||   __|  |  |     |  |
// |  `----.|  '--'  ||  |     |  `----.|  |
//  \______||_______/ |__|      \______||__|
//
// WTPM wavefunction storage — optimized inline-array layout
// Based on: Gao, Li, Shen, SIAM J. Sci. Comput., 46(1), A179-A203, 2024

#ifndef CDFCI_WTPM_WAVEFUNCTION_H
#define CDFCI_WTPM_WAVEFUNCTION_H 1

#include <cmath>
#include <array>
#include <vector>
#include "hash.h"
#include "lib/libcuckoo/cuckoohash_map.hh"

constexpr int WTPM_MAX_P = 32;

template<int N = 1>
class WTPMWaveFunctionVector
{
    public:

    using value_type = std::pair<Determinant<N>, std::array<double, 2>>;
    std::vector<value_type> data;

    size_t n_new_element = 0;
    double new_z = 0;

    WTPMWaveFunctionVector() {}

    ~WTPMWaveFunctionVector() {}

    size_t size() const { return data.size(); }

    void clear()
    {
        data.clear();
        n_new_element = 0;
        new_z = 0;
    }

    void append(WTPMWaveFunctionVector<N>& sub)
    {
        data.insert(data.end(), sub.data.begin(), sub.data.end());
    }
};

template<int N = 1>
class WTPMWaveFunctionBase
{
    public:

    using key_type = Determinant<N>;
    int p;
    double mu;

    std::array<double, WTPM_MAX_P * WTPM_MAX_P> S;
    std::array<double, WTPM_MAX_P> d;
    std::array<double, WTPM_MAX_P> W;

    size_t max_size = 0;

    WTPMWaveFunctionBase(int num_eigenpairs = 1, double penalty = 1.0)
        : p(num_eigenpairs), mu(penalty)
    {
        S.fill(0.0);
        d.fill(0.0);
        W.fill(0.0);
    }

    ~WTPMWaveFunctionBase() {}

    double S_ref(int i, int j) const { return S[i * WTPM_MAX_P + j]; }
    double& S_ref(int i, int j) { return S[i * WTPM_MAX_P + j]; }

    double get_energy(int l) const
    {
        return d[l] / S_ref(l, l);
    }

    void update_d(int l, double alpha, double Y_new_kl, double A_kk)
    {
        d[l] += 2.0 * alpha * Y_new_kl - alpha * alpha * A_kk;
    }
};

template<int N = 1>
class WTPMWaveFunctionStd : public WTPMWaveFunctionBase<N>
{
    public:

    using typename WTPMWaveFunctionBase<N>::key_type;
    using WTPMWaveFunctionBase<N>::p;
    using WTPMWaveFunctionBase<N>::S_ref;
    using WTPMWaveFunctionBase<N>::d;
    using WTPMWaveFunctionBase<N>::W;
    using WTPMWaveFunctionBase<N>::max_size;
    using value_storage = std::array<double, 2 * WTPM_MAX_P>;

    WTPMWaveFunctionStd(size_t capacity, int num_eigenpairs = 1, double penalty = 1.0)
        : WTPMWaveFunctionBase<N>(num_eigenpairs, penalty)
    {
        data_.reserve(capacity);
        max_size = capacity;
    }

    ~WTPMWaveFunctionStd() {}

    size_t size() { return data_.size(); }

    void update_x(key_type& key, int l, double dx, const double* x_k_all)
    {
        auto iter = data_.find(key);
        if (iter != data_.end())
        {
            auto& val = iter->second;
            double x_old = x_k_all[l];

            val[l] += dx;

            S_ref(l, l) += 2.0 * dx * x_old + dx * dx;
            for (int i = 0; i < p; ++i)
            {
                if (i != l)
                {
                    S_ref(i, l) += dx * x_k_all[i];
                    S_ref(l, i) = S_ref(i, l);
                }
            }
        }
        else
        {
            value_storage val{};
            val[l] = dx;
            data_.insert({key, val});
            S_ref(l, l) += dx * dx;
        }
    }

    void update_z_only(std::vector<std::pair<key_type, double>>& column,
        double dx, int l, WTPMWaveFunctionVector<N>& sub_xz, double z_threshold = 0.0)
    {
        sub_xz.new_z = 0;

        for (auto& entry : column)
        {
            auto& det = entry.first;
            double h = entry.second;
            double dz = dx * h;

            auto iter = data_.find(det);
            if (iter != data_.end())
            {
                auto& val = iter->second;
                val[p + l] += dz;
                sub_xz.data.push_back({det, {val[l], val[p + l]}});
                sub_xz.new_z += h * val[l];
            }
            else
            {
                if (fabs(dz) > z_threshold)
                {
                    value_storage new_val{};
                    new_val[p + l] = dz;
                    data_.insert({det, new_val});
                    sub_xz.data.push_back({det, {0.0, dz}});
                }
            }
        }
    }

    void update_z_and_get_sub(std::vector<std::pair<key_type, double>>& column,
        double dx, int l, double A_kk, WTPMWaveFunctionVector<N>& sub_xz, double z_threshold = 0.0)
    {
        sub_xz.clear();
        update_z_only(column, dx, l, sub_xz, z_threshold);
    }

    void reinsert_z(key_type& key, int l, double new_z)
    {
        auto iter = data_.find(key);
        if (iter != data_.end())
        {
            iter->second[p + l] = new_z;
        }
    }

    void get_x(key_type& key, double* out) const
    {
        auto iter = data_.find(key);
        if (iter != data_.end())
        {
            const auto& val = iter->second;
            for (int i = 0; i < p; ++i) out[i] = val[i];
        }
        else
        {
            for (int i = 0; i < p; ++i) out[i] = 0.0;
        }
    }

    bool get_xz(key_type& key, int l, double& x_val, double& z_val) const
    {
        auto iter = data_.find(key);
        if (iter != data_.end())
        {
            const auto& val = iter->second;
            x_val = val[l];
            z_val = val[p + l];
            return true;
        }
        x_val = 0.0;
        z_val = 0.0;
        return false;
    }

    bool get_x_and_xz(key_type& key, int l, double* x_all, double& x_val, double& z_val) const
    {
        auto iter = data_.find(key);
        if (iter != data_.end())
        {
            const auto& val = iter->second;
            for (int i = 0; i < p; ++i) x_all[i] = val[i];
            x_val = val[l];
            z_val = val[p + l];
            return true;
        }
        for (int i = 0; i < p; ++i) x_all[i] = 0.0;
        x_val = 0.0;
        z_val = 0.0;
        return false;
    }

    private:

    robin_hood::unordered_flat_map<key_type, value_storage,
        DeterminantHash<N>, DeterminantEqual<N>> data_;
};

template<int N = 1>
class WTPMWaveFunctionCuckoo : public WTPMWaveFunctionBase<N>
{
    public:

    using typename WTPMWaveFunctionBase<N>::key_type;
    using WTPMWaveFunctionBase<N>::p;
    using WTPMWaveFunctionBase<N>::S_ref;
    using WTPMWaveFunctionBase<N>::d;
    using WTPMWaveFunctionBase<N>::W;
    using WTPMWaveFunctionBase<N>::max_size;
    using value_storage = std::array<double, 2 * WTPM_MAX_P>;

    WTPMWaveFunctionCuckoo(size_t capacity, int num_eigenpairs = 1, double penalty = 1.0)
        : WTPMWaveFunctionBase<N>(num_eigenpairs, penalty)
    {
        data_.reserve(capacity);
        max_size = capacity;
    }

    ~WTPMWaveFunctionCuckoo() {}

    size_t size() { return data_.size(); }

    void update_x(key_type& key, int l, double dx, const double* x_k_all)
    {
        value_storage old_val{};
        value_storage val_new{};
        val_new[l] = dx;

        auto update_fn = [l, dx, &old_val](value_storage& val) {
            old_val = val;
            val[l] += dx;
            return false;
        };
        data_.upsert(key, update_fn, val_new);

        double x_old = old_val[l];

        S_ref(l, l) += 2.0 * dx * x_old + dx * dx;
        for (int i = 0; i < p; ++i)
        {
            if (i != l)
            {
                S_ref(i, l) += dx * x_k_all[i];
                S_ref(l, i) = S_ref(i, l);
            }
        }
    }

    void update_z_only(std::vector<std::pair<key_type, double>>& column,
        double dx, int l, WTPMWaveFunctionVector<N>& sub_xz, double z_threshold = 0.0)
    {
        sub_xz.new_z = 0;
        sub_xz.n_new_element = 0;

        for (auto& entry : column)
        {
            auto& det = entry.first;
            double h = entry.second;
            double dz = dx * h;

            double x_val = 0, z_val = 0;
            auto update_functor = [dz, l, p=this->p, &x_val, &z_val](value_storage& val) {
                val[p + l] += dz;
                x_val = val[l];
                z_val = val[p + l];
                return false;
            };

            auto flag_found = data_.update_fn(det, update_functor);
            if (flag_found)
            {
                sub_xz.data.push_back({det, {x_val, z_val}});
                sub_xz.new_z += h * x_val;
            }
            else
            {
                if (fabs(dz) > z_threshold)
                {
                    value_storage new_val{};
                    new_val[p + l] = dz;
                    data_.insert(det, new_val);
                    sub_xz.data.push_back({det, {0.0, dz}});
                    sub_xz.n_new_element += 1;
                }
            }
        }

        if (size() > 0.79 * max_size)
        {
            throw std::overflow_error("The hash table is full. Please increase max_wavefunction_size or z_threshold.");
        }
    }

    void update_z_and_get_sub(std::vector<std::pair<key_type, double>>& column,
        double dx, int l, double A_kk, WTPMWaveFunctionVector<N>& sub_xz, double z_threshold = 0.0)
    {
        sub_xz.clear();
        update_z_only(column, dx, l, sub_xz, z_threshold);
    }

    void reinsert_z(key_type& key, int l, double new_z)
    {
        auto update_fn = [l, new_z, p=this->p](value_storage& val) {
            val[p + l] = new_z;
            return false;
        };
        data_.upsert(key, update_fn, value_storage{});
    }

    void get_x(key_type& key, double* out) const
    {
        bool found = data_.find_fn(key, [p=this->p, out](const value_storage& val) {
            for (int i = 0; i < p; ++i) out[i] = val[i];
        });
        if (!found) {
            for (int i = 0; i < p; ++i) out[i] = 0.0;
        }
    }

    bool get_xz(key_type& key, int l, double& x_val, double& z_val) const
    {
        bool found = data_.find_fn(key, [l, p=this->p, &x_val, &z_val](const value_storage& val) {
            x_val = val[l];
            z_val = val[p + l];
        });
        if (!found) { x_val = 0.0; z_val = 0.0; }
        return found;
    }

    bool get_x_and_xz(key_type& key, int l, double* x_all, double& x_val, double& z_val) const
    {
        bool found = data_.find_fn(key, [l, p=this->p, x_all, &x_val, &z_val](const value_storage& val) {
            for (int i = 0; i < p; ++i) x_all[i] = val[i];
            x_val = val[l];
            z_val = val[p + l];
        });
        if (!found) {
            for (int i = 0; i < p; ++i) x_all[i] = 0.0;
            x_val = 0.0;
            z_val = 0.0;
        }
        return found;
    }

    private:

    cuckoohash_map<key_type, value_storage,
        DeterminantHashRobinhood<N>, DeterminantEqual<N>,
        std::allocator<std::pair<const key_type, value_storage>>, 8> data_;
};

#endif
