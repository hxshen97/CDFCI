//   ______  _______   _______   ______  __
//  /      ||       \ |   ____| /      ||  |
// |  ,----'|  .--.  ||  |__   |  ,----'|  |
// |  |     |  |  |  ||   __|  |  |     |  |
// |  `----.|  '--'  ||  |     |  `----.|  |
//  \______||_______/ |__|      \______||__|
//
// WTPM-CD orchestration wrapper

#ifndef CDFCI_WTPM_WRAPPER_H
#define CDFCI_WTPM_WRAPPER_H 1

#include "wtpm_solver.h"

Option read_wtpm_input(const std::string input_file)
{
    // Default WTPM options
    Option default_wtpm_option = {
        {"p", 1},
        {"mu", 1.0},
        {"num_iterations", 30000},
        {"report_interval", 1000},
        {"z_threshold", 0.0},
        {"z_threshold_search", false},
        {"max_wavefunction_size", 1},
        {"max_memory", 0.0}
    };
    Option default_hamiltonian_option = {
        {"type", "molecule"},
        {"threshold", 1e-13}
    };

    Option option = option_from_file(input_file);

    default_wtpm_option.merge_patch(option["solver"]["wtpm"]);
    option["solver"]["wtpm"] = default_wtpm_option;
    default_hamiltonian_option.merge_patch(option["hamiltonian"]);
    option["hamiltonian"] = default_hamiltonian_option;

    // Validate
    if (option["hamiltonian"]["type"] != "molecule")
    {
        throw std::invalid_argument("hamiltonian:type is invalid. Currently the only supported "
         "Hamiltonian is \"molecule\".");
    }
    double max_memory = option["solver"]["wtpm"]["max_memory"];
    if (max_memory <= 0.0)
    {
        throw std::invalid_argument("solver:wtpm:max_memory is invalid. It is the maximum memory (GB)"
            " allowed for the wavefunction. Please set a positive number.");
    }
    int p = option["solver"]["wtpm"]["p"];
    if (p < 1)
    {
        throw std::invalid_argument("solver:wtpm:p must be a positive integer.");
    }

    return option;
}

template <int N>
std::vector<double> run_wtpm(Option& option, Fcidump& fci)
{
    // Step 1: Initialize Hamiltonian
    HamiltonianMolecule<N> h(fci, option["hamiltonian"]["threshold"]);

    // Step 2: Initialize solver
    Option wtpm_option = option["solver"]["wtpm"];
    WTPMCD<N> wtpm_solver(wtpm_option);
    Determinant<N>::constuct_masks();

    // Step 3: Solve
    wtpm_solver.solve(h);

    // Collect energies
    std::vector<double> energies;
    for (int l = 0; l < wtpm_solver.p; ++l)
    {
        energies.push_back(wtpm_solver.result_energies[l]);
    }
    return energies;
}

std::vector<double> run_wtpm_wrapper(Option& option)
{
    // Read FCIDUMP
    Fcidump fci(option["hamiltonian"]["fcidump_path"]);

    int n = 1 + (fci.norb - 1) / (sizeof(Determinant<1>::det_type) * CHAR_BIT);
    int p = option["solver"]["wtpm"]["p"];

    // Print basic information
    std::cout << "FCIDUMP information" << std::endl;
    std::cout << "-------------------" << std::endl;
    std::cout << "Number of electrons: " << fci.nelec << std::endl;
    std::cout << "Number of spin-orbitals: " << fci.norb << std::endl;
    std::cout << "Ms2: " << fci.ms2 << std::endl;
    std::string hf_type = fci.uhf ? "unrestricted" : "restricted";
    std::cout << "Hartree Fock: " + hf_type << std::endl << std::endl;

    std::cout << "Machine information" << std::endl;
    std::cout << "-------------------" << std::endl;
    std::cout << "sizeof(size_t): " << sizeof(Determinant<1>::det_type) << " bytes or ";
    std::cout << (sizeof(Determinant<1>::det_type) * CHAR_BIT) << " bits" << std::endl;
    std::cout << "Number of size_t to represent one Slater determinant: " << n << std::endl;
#ifdef CDFCI_USE_LONG_DOUBLE
    std::cout << "__float128: disabled" << std::endl << std::endl;
#else
    std::cout << "__float128: enabled" << std::endl << std::endl;
#endif

    // Compute max_wavefunction_size
    // Each determinant: n * sizeof(size_t) + 2*WTPM_MAX_P * sizeof(double)
    int bytes_per_det = n * sizeof(Determinant<1>::det_type) + 2 * WTPM_MAX_P * sizeof(double);
    size_t max_wavefunction_size = 8;
    double max_load_factor = 0.79;
    double max_memory = option["solver"]["wtpm"]["max_memory"];
    while (max_wavefunction_size * bytes_per_det < max_memory * 1073741824)
    {
        max_wavefunction_size *= 2;
    }
    max_wavefunction_size /= 2;
    option["solver"]["wtpm"]["max_wavefunction_size"] = static_cast<size_t>(max_load_factor * max_wavefunction_size);

    std::cout << "Input option" << std::endl;
    std::cout << "------------" << std::endl;
    std::cout << option.dump(4) << std::endl;
    std::cout << "Note:" << std::endl;
    std::cout << "In the WTPM wavefunction, each determinant needs " << bytes_per_det << " bytes." << std::endl;
    std::cout << "The size of the wavefunction is 2^" << static_cast<int>(log2(max_wavefunction_size))
              << " = " << max_wavefunction_size
              << ", which can store " << static_cast<size_t>(max_load_factor * max_wavefunction_size)
              << " determinants at most." << std::endl;
    std::cout << "The wavefunction uses about " << std::fixed << std::setprecision(3)
              << static_cast<double>(max_wavefunction_size) * bytes_per_det / 1073741824
              << " GB memory." << std::endl << std::endl;

    // Run WTPM
    switch (n)
    {
    case 1:
        return run_wtpm<1>(option, fci);
    case 2:
        return run_wtpm<2>(option, fci);
    case 3:
        return run_wtpm<3>(option, fci);
    case 4:
        return run_wtpm<4>(option, fci);
    default:
        throw std::invalid_argument("The CDFCI program only supports 1 - 4 size_t integers for a Slater determinant.");
    }
}

#endif
