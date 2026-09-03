//=============================================================================
//
//   Copyright (c) by Computer Graphics Group, Bielefeld University
//
// This work is licensed under a
// Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
//
// You should have received a copy of the license along with this
// work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
//
//=============================================================================

#include "MultilinearModel.h"
#include "utils.h"

#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>
#include <string>

#include <pmp/SurfaceMesh.h>

//=============================================================================

namespace
{

std::string with_trailing_separator(const std::string& path)
{
    if (path.empty())
        return path;

    const char last = path[path.size() - 1];
    if (last == '/' || last == '\\')
        return path;

    return path + "/";
}

void print_usage(const char* executable)
{
    std::cerr
        << "Usage:\n"
        << "  " << executable << " --model-dir <dir> --out-skin <file> --out-skull <file> [options]\n\n"
        << "Options:\n"
        << "  --model-dir <dir>   Directory containing skin.off, skull.off, and MLM data files.\n"
        << "  --out-skin <file>   Output skin mesh path, for example fitted_skin.off.\n"
        << "  --out-skull <file>  Output skull mesh path, for example estimated_skull.off.\n"
        << "  --w-skull <file>    Optional skull parameter .scalars file.\n"
        << "  --w-fstt <file>     Optional FSTT parameter .scalars file.\n"
        << "  --out-w-skull <file> Optional output file for the parameters used.\n"
        << "  --out-w-fstt <file> Optional output file for the parameters used.\n"
        << "  --help              Show this help message.\n";
}

bool save_vector(const Eigen::VectorXd& values, const std::string& filename)
{
    std::ofstream ofs(filename.c_str());
    if (!ofs.is_open())
    {
        std::cerr << "Cannot write parameters to " << filename << std::endl;
        return false;
    }

    for (int i = 0; i < values.size(); ++i)
        ofs << values(i) << "\n";

    return true;
}

void initialize_mean_parameters(
    Eigen::VectorXd& w_skull,
    Eigen::VectorXd& w_fstt,
    const MultilinearModel& model)
{
    w_skull = Eigen::VectorXd::Zero(model.dim1());
    for (int i = 0; i < model.U_skull().rows(); ++i)
    {
        for (int j = 0; j < model.U_skull().cols(); ++j)
            w_skull(j) += model.U_skull()(i, j);
    }
    w_skull /= static_cast<double>(model.U_skull().rows());

    w_fstt = Eigen::VectorXd::Zero(model.dim2());
    for (int i = 0; i < model.U_fstt().rows(); ++i)
    {
        for (int j = 0; j < model.U_fstt().cols(); ++j)
            w_fstt(j) += model.U_fstt()(i, j);
    }
    w_fstt /= static_cast<double>(model.U_fstt().rows());
}

} // namespace

//=============================================================================

int main(int argc, char** argv)
{
    std::string model_dir = "../data/";
    std::string output_skin;
    std::string output_skull;
    std::string input_w_skull;
    std::string input_w_fstt;
    std::string output_w_skull;
    std::string output_w_fstt;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        else if (arg == "--model-dir" && i + 1 < argc)
        {
            model_dir = argv[++i];
        }
        else if (arg == "--out-skin" && i + 1 < argc)
        {
            output_skin = argv[++i];
        }
        else if (arg == "--out-skull" && i + 1 < argc)
        {
            output_skull = argv[++i];
        }
        else if (arg == "--w-skull" && i + 1 < argc)
        {
            input_w_skull = argv[++i];
        }
        else if (arg == "--w-fstt" && i + 1 < argc)
        {
            input_w_fstt = argv[++i];
        }
        else if (arg == "--out-w-skull" && i + 1 < argc)
        {
            output_w_skull = argv[++i];
        }
        else if (arg == "--out-w-fstt" && i + 1 < argc)
        {
            output_w_fstt = argv[++i];
        }
        else
        {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (output_skin.empty() || output_skull.empty())
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (input_w_skull.empty() != input_w_fstt.empty())
    {
        std::cerr << "--w-skull and --w-fstt must be provided together." << std::endl;
        return EXIT_FAILURE;
    }

    model_dir = with_trailing_separator(model_dir);
    const std::string filename_skin = model_dir + "skin.off";
    const std::string filename_skull = model_dir + "skull.off";

    pmp::SurfaceMesh skin;
    pmp::SurfaceMesh skull;
    if (!(skin.read(filename_skin) && skull.read(filename_skull)))
    {
        std::cerr << "Cannot load skin and skull meshes from " << model_dir << std::endl;
        return EXIT_FAILURE;
    }

    MultilinearModel model;
    if (!model.load_means(filename_skin, filename_skull))
    {
        std::cerr << "Cannot load mean skin/skull meshes." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Loading multilinear model from " << model_dir << " ..." << std::flush;
    if (!model.load(model_dir))
    {
        std::cerr << "\nCannot load multilinear model." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << " done." << std::endl;

    Eigen::VectorXd w_skull;
    Eigen::VectorXd w_fstt;
    initialize_mean_parameters(w_skull, w_fstt, model);

    if (!input_w_skull.empty())
    {
        if (!load_parameters(w_skull, w_fstt, input_w_skull, input_w_fstt))
            return EXIT_FAILURE;
    }

    if (!model.evaluate(skin, skull, w_skull, w_fstt))
    {
        std::cerr << "Cannot evaluate multilinear model." << std::endl;
        return EXIT_FAILURE;
    }

    if (!skin.write(output_skin))
    {
        std::cerr << "Cannot write skin mesh to " << output_skin << std::endl;
        return EXIT_FAILURE;
    }

    if (!skull.write(output_skull))
    {
        std::cerr << "Cannot write skull mesh to " << output_skull << std::endl;
        return EXIT_FAILURE;
    }

    if (!output_w_skull.empty() && !save_vector(w_skull, output_w_skull))
        return EXIT_FAILURE;

    if (!output_w_fstt.empty() && !save_vector(w_fstt, output_w_fstt))
        return EXIT_FAILURE;

    std::cout << "Wrote skin mesh: " << output_skin << std::endl;
    std::cout << "Wrote skull mesh: " << output_skull << std::endl;
    return EXIT_SUCCESS;
}

//=============================================================================
