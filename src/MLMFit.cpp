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

#include <pmp/BoundingBox.h>
#include <pmp/SurfaceMesh.h>
#include <pmp/algorithms/TriangleKdTree.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

//=============================================================================

namespace
{

struct Options
{
    std::string model_dir = "../data/";
    std::string target;
    std::string output_skin;
    std::string output_skull;
    std::string output_w_skull;
    std::string output_w_fstt;
    std::string report;
    std::string input_w_skull;
    std::string input_w_fstt;

    int iterations = 60;
    int sample_stride = 8;
    double initial_step = 0.075;
    double min_step = 0.001;
    double step_decay = 0.5;
    double max_distance = 20.0;
    double prior_weight = 0.001;
    bool align_centers = true;
    bool scale_target_to_model = false;
};

struct FitState
{
    double loss = DBL_MAX;
    double data_loss = DBL_MAX;
    double prior_loss = 0.0;
    double mean_distance = 0.0;
    double rms_distance = 0.0;
    double max_observed_distance = 0.0;
    int sample_count = 0;
};

void print_usage(const char* executable)
{
    std::cerr
        << "Usage:\n"
        << "  " << executable << " --target <mesh-or-points> --out-skin <file> --out-skull <file> [options]\n\n"
        << "Options:\n"
        << "  --model-dir <dir>        Directory containing skin.off, skull.off, and MLM data files.\n"
        << "  --target <file>          Target MICA mesh or point set readable by PMP.\n"
        << "  --out-skin <file>        Output fitted skin mesh path.\n"
        << "  --out-skull <file>       Output estimated skull mesh path.\n"
        << "  --out-w-skull <file>     Optional output skull parameter .scalars file.\n"
        << "  --out-w-fstt <file>      Optional output FSTT parameter .scalars file.\n"
        << "  --report <file>          Optional text report path.\n"
        << "  --w-skull <file>         Optional initial skull parameter .scalars file.\n"
        << "  --w-fstt <file>          Optional initial FSTT parameter .scalars file.\n"
        << "  --iterations <n>         Coordinate-search iterations. Default: 60.\n"
        << "  --sample-stride <n>      Use every n-th model skin vertex. Default: 8.\n"
        << "  --initial-step <value>   Initial parameter step. Default: 0.075.\n"
        << "  --min-step <value>       Stop after step drops below this value. Default: 0.001.\n"
        << "  --step-decay <value>     Step multiplier when no parameter improves. Default: 0.5.\n"
        << "  --max-distance <value>   Clamp correspondence distances. Use 0 to disable. Default: 20.\n"
        << "  --prior-weight <value>   L2 regularization toward initial parameters. Default: 0.001.\n"
        << "  --no-align-centers       Do not translate target to the initial model center.\n"
        << "  --scale-target-to-model  Uniformly scale target bounding box size to initial model size.\n"
        << "  --help                   Show this help message.\n";
}

bool read_int_arg(int& value, int& index, int argc, char** argv)
{
    if (index + 1 >= argc)
        return false;
    value = std::atoi(argv[++index]);
    return true;
}

bool read_double_arg(double& value, int& index, int argc, char** argv)
{
    if (index + 1 >= argc)
        return false;
    value = std::atof(argv[++index]);
    return true;
}

bool parse_args(Options& options, int argc, char** argv)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            return false;
        }
        else if (arg == "--model-dir" && i + 1 < argc)
        {
            options.model_dir = argv[++i];
        }
        else if (arg == "--target" && i + 1 < argc)
        {
            options.target = argv[++i];
        }
        else if (arg == "--out-skin" && i + 1 < argc)
        {
            options.output_skin = argv[++i];
        }
        else if (arg == "--out-skull" && i + 1 < argc)
        {
            options.output_skull = argv[++i];
        }
        else if (arg == "--out-w-skull" && i + 1 < argc)
        {
            options.output_w_skull = argv[++i];
        }
        else if (arg == "--out-w-fstt" && i + 1 < argc)
        {
            options.output_w_fstt = argv[++i];
        }
        else if (arg == "--report" && i + 1 < argc)
        {
            options.report = argv[++i];
        }
        else if (arg == "--w-skull" && i + 1 < argc)
        {
            options.input_w_skull = argv[++i];
        }
        else if (arg == "--w-fstt" && i + 1 < argc)
        {
            options.input_w_fstt = argv[++i];
        }
        else if (arg == "--iterations")
        {
            if (!read_int_arg(options.iterations, i, argc, argv))
                return false;
        }
        else if (arg == "--sample-stride")
        {
            if (!read_int_arg(options.sample_stride, i, argc, argv))
                return false;
        }
        else if (arg == "--initial-step")
        {
            if (!read_double_arg(options.initial_step, i, argc, argv))
                return false;
        }
        else if (arg == "--min-step")
        {
            if (!read_double_arg(options.min_step, i, argc, argv))
                return false;
        }
        else if (arg == "--step-decay")
        {
            if (!read_double_arg(options.step_decay, i, argc, argv))
                return false;
        }
        else if (arg == "--max-distance")
        {
            if (!read_double_arg(options.max_distance, i, argc, argv))
                return false;
        }
        else if (arg == "--prior-weight")
        {
            if (!read_double_arg(options.prior_weight, i, argc, argv))
                return false;
        }
        else if (arg == "--no-align-centers")
        {
            options.align_centers = false;
        }
        else if (arg == "--scale-target-to-model")
        {
            options.scale_target_to_model = true;
        }
        else
        {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            print_usage(argv[0]);
            return false;
        }
    }

    if (options.target.empty() || options.output_skin.empty() || options.output_skull.empty())
    {
        print_usage(argv[0]);
        return false;
    }

    if (options.input_w_skull.empty() != options.input_w_fstt.empty())
    {
        std::cerr << "--w-skull and --w-fstt must be provided together." << std::endl;
        return false;
    }

    options.iterations = std::max(1, options.iterations);
    options.sample_stride = std::max(1, options.sample_stride);
    options.initial_step = std::max(1e-12, options.initial_step);
    options.min_step = std::max(1e-12, options.min_step);
    options.step_decay = std::min(0.99, std::max(0.01, options.step_decay));
    options.prior_weight = std::max(0.0, options.prior_weight);

    return true;
}

pmp::Point transform_point(
    const pmp::Point& point,
    const pmp::Point& target_center,
    const pmp::Point& model_center,
    double scale,
    bool align_centers)
{
    pmp::Point output = point;
    if (align_centers)
        output = model_center + static_cast<pmp::Scalar>(scale) * (point - target_center);
    return output;
}

void transform_target_mesh(
    pmp::SurfaceMesh& target,
    const pmp::Point& model_center,
    double model_size,
    const Options& options)
{
    if (!options.align_centers && !options.scale_target_to_model)
        return;

    const pmp::BoundingBox target_bounds = target.bounds();
    const pmp::Point target_center = target_bounds.center();
    const double target_size = static_cast<double>(target_bounds.size());
    double scale = 1.0;
    if (options.scale_target_to_model && target_size > 1e-12)
        scale = model_size / target_size;

    auto target_points = target.vertex_property<pmp::Point>("v:point");
    for (auto v : target.vertices())
        target_points[v] = transform_point(target_points[v], target_center, model_center, scale, options.align_centers);

    if (options.align_centers)
        std::cout << "Aligned target center to initial model center." << std::endl;
    if (options.scale_target_to_model)
        std::cout << "Scaled target to model bounding-box size with factor " << scale << "." << std::endl;
}

std::vector<pmp::Point> collect_target_points(const pmp::SurfaceMesh& target)
{
    std::vector<pmp::Point> points;
    points.reserve(target.n_vertices());
    auto target_points = target.get_vertex_property<pmp::Point>("v:point");
    for (auto v : target.vertices())
        points.push_back(target_points[v]);
    return points;
}

double nearest_vertex_distance(
    const pmp::Point& point,
    const std::vector<pmp::Point>& target_points)
{
    double best = DBL_MAX;
    for (const pmp::Point& target_point : target_points)
    {
        const double distance = static_cast<double>(pmp::distance(point, target_point));
        if (distance < best)
            best = distance;
    }
    return best;
}

FitState compute_loss(
    const MultilinearModel& model,
    pmp::SurfaceMesh& skin,
    pmp::SurfaceMesh& skull,
    const pmp::TriangleKdTree* target_tree,
    const std::vector<pmp::Point>& target_points,
    const Eigen::VectorXd& w_skull,
    const Eigen::VectorXd& w_fstt,
    const Eigen::VectorXd& w_skull_initial,
    const Eigen::VectorXd& w_fstt_initial,
    const Options& options)
{
    model.evaluate(skin, skull, w_skull, w_fstt);

    FitState state;
    state.data_loss = 0.0;
    state.sample_count = 0;

    const double max_distance = options.max_distance;
    auto skin_points = skin.vertex_property<pmp::Point>("v:point");
    int vertex_index = 0;
    for (auto v : skin.vertices())
    {
        if ((vertex_index++ % options.sample_stride) != 0)
            continue;

        double distance = 0.0;
        if (target_tree)
        {
            const pmp::TriangleKdTree::NearestNeighbor nn = target_tree->nearest(skin_points[v]);
            distance = static_cast<double>(nn.dist);
        }
        else
        {
            distance = nearest_vertex_distance(skin_points[v], target_points);
        }

        state.mean_distance += distance;
        state.rms_distance += distance * distance;
        state.max_observed_distance = std::max(state.max_observed_distance, distance);

        if (max_distance > 0.0)
            distance = std::min(distance, max_distance);

        state.data_loss += distance * distance;
        ++state.sample_count;
    }

    if (state.sample_count > 0)
    {
        state.data_loss /= static_cast<double>(state.sample_count);
        state.mean_distance /= static_cast<double>(state.sample_count);
        state.rms_distance = std::sqrt(state.rms_distance / static_cast<double>(state.sample_count));
    }

    state.prior_loss =
        (w_skull - w_skull_initial).squaredNorm() / static_cast<double>(std::max(1, static_cast<int>(w_skull.size()))) +
        (w_fstt - w_fstt_initial).squaredNorm() / static_cast<double>(std::max(1, static_cast<int>(w_fstt.size())));
    state.loss = state.data_loss + options.prior_weight * state.prior_loss;

    return state;
}

void write_report(
    const std::string& filename,
    const Options& options,
    const FitState& state,
    const Eigen::VectorXd& w_skull,
    const Eigen::VectorXd& w_fstt)
{
    if (filename.empty())
        return;

    std::ofstream ofs(filename.c_str());
    if (!ofs.is_open())
    {
        std::cerr << "Cannot write report to " << filename << std::endl;
        return;
    }

    ofs << "target: " << options.target << "\n";
    ofs << "model_dir: " << options.model_dir << "\n";
    ofs << "iterations: " << options.iterations << "\n";
    ofs << "sample_stride: " << options.sample_stride << "\n";
    ofs << "initial_step: " << options.initial_step << "\n";
    ofs << "min_step: " << options.min_step << "\n";
    ofs << "step_decay: " << options.step_decay << "\n";
    ofs << "max_distance: " << options.max_distance << "\n";
    ofs << "prior_weight: " << options.prior_weight << "\n";
    ofs << "loss: " << state.loss << "\n";
    ofs << "data_loss: " << state.data_loss << "\n";
    ofs << "prior_loss: " << state.prior_loss << "\n";
    ofs << "mean_distance: " << state.mean_distance << "\n";
    ofs << "rms_distance: " << state.rms_distance << "\n";
    ofs << "max_observed_distance: " << state.max_observed_distance << "\n";
    ofs << "sample_count: " << state.sample_count << "\n";
    ofs << "w_skull:";
    for (int i = 0; i < w_skull.size(); ++i)
        ofs << " " << w_skull(i);
    ofs << "\nw_fstt:";
    for (int i = 0; i < w_fstt.size(); ++i)
        ofs << " " << w_fstt(i);
    ofs << "\n";
}

} // namespace

//=============================================================================

int main(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }

    Options options;
    if (!parse_args(options, argc, argv))
        return EXIT_FAILURE;

    options.model_dir = with_trailing_separator(options.model_dir);
    const std::string filename_skin = options.model_dir + "skin.off";
    const std::string filename_skull = options.model_dir + "skull.off";

    pmp::SurfaceMesh skin;
    pmp::SurfaceMesh skull;
    if (!(skin.read(filename_skin) && skull.read(filename_skull)))
    {
        std::cerr << "Cannot load skin and skull meshes from " << options.model_dir << std::endl;
        return EXIT_FAILURE;
    }

    MultilinearModel model;
    if (!model.load_means(filename_skin, filename_skull))
    {
        std::cerr << "Cannot load mean skin/skull meshes." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Loading multilinear model from " << options.model_dir << " ..." << std::flush;
    if (!model.load(options.model_dir))
    {
        std::cerr << "\nCannot load multilinear model." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << " done." << std::endl;

    Eigen::VectorXd w_skull;
    Eigen::VectorXd w_fstt;
    initialize_mean_parameters(w_skull, w_fstt, model);
    const Eigen::VectorXd w_skull_initial = w_skull;
    const Eigen::VectorXd w_fstt_initial = w_fstt;

    if (!options.input_w_skull.empty())
    {
        if (!load_parameters(w_skull, w_fstt, options.input_w_skull, options.input_w_fstt))
            return EXIT_FAILURE;
    }

    model.evaluate(skin, skull, w_skull, w_fstt);
    const pmp::BoundingBox model_bounds = skin.bounds();

    pmp::SurfaceMesh target;
    if (!target.read(options.target))
    {
        std::cerr << "Cannot load target mesh or point set: " << options.target << std::endl;
        return EXIT_FAILURE;
    }
    if (target.n_vertices() == 0)
    {
        std::cerr << "Target has no vertices: " << options.target << std::endl;
        return EXIT_FAILURE;
    }

    transform_target_mesh(target, model_bounds.center(), static_cast<double>(model_bounds.size()), options);

    std::unique_ptr<pmp::TriangleKdTree> target_tree;
    std::vector<pmp::Point> target_points;
    if (target.n_faces() > 0)
    {
        target_tree.reset(new pmp::TriangleKdTree(target));
        std::cout << "Using target triangle surface: "
                  << target.n_vertices() << " vertices, "
                  << target.n_faces() << " faces." << std::endl;
    }
    else
    {
        target_points = collect_target_points(target);
        std::cout << "Using target vertices as a point set: "
                  << target_points.size() << " points." << std::endl;
    }

    FitState best = compute_loss(
        model, skin, skull, target_tree.get(), target_points,
        w_skull, w_fstt, w_skull_initial, w_fstt_initial, options);

    double step = options.initial_step;
    std::cout << std::fixed << std::setprecision(6)
              << "Initial loss: " << best.loss
              << ", RMS distance: " << best.rms_distance << std::endl;

    for (int iteration = 0; iteration < options.iterations && step >= options.min_step; ++iteration)
    {
        bool improved = false;

        for (int block = 0; block < 2; ++block)
        {
            Eigen::VectorXd& weights = block == 0 ? w_skull : w_fstt;
            const int count = static_cast<int>(weights.size());
            for (int index = 0; index < count; ++index)
            {
                const double original = weights(index);
                double best_value = original;
                FitState best_candidate = best;

                for (const double direction : {1.0, -1.0})
                {
                    weights(index) = original + direction * step;
                    const FitState candidate = compute_loss(
                        model, skin, skull, target_tree.get(), target_points,
                        w_skull, w_fstt, w_skull_initial, w_fstt_initial, options);

                    if (candidate.loss < best_candidate.loss)
                    {
                        best_candidate = candidate;
                        best_value = weights(index);
                    }
                }

                weights(index) = best_value;
                if (best_candidate.loss < best.loss)
                {
                    best = best_candidate;
                    improved = true;
                }
            }
        }

        if (!improved)
            step *= options.step_decay;

        std::cout << "Iteration " << (iteration + 1)
                  << " loss=" << best.loss
                  << " rms=" << best.rms_distance
                  << " step=" << step
                  << (improved ? " improved" : " reduced-step")
                  << std::endl;
    }

    best = compute_loss(
        model, skin, skull, target_tree.get(), target_points,
        w_skull, w_fstt, w_skull_initial, w_fstt_initial, options);

    if (!skin.write(options.output_skin))
    {
        std::cerr << "Cannot write fitted skin mesh to " << options.output_skin << std::endl;
        return EXIT_FAILURE;
    }

    if (!skull.write(options.output_skull))
    {
        std::cerr << "Cannot write estimated skull mesh to " << options.output_skull << std::endl;
        return EXIT_FAILURE;
    }

    if (!options.output_w_skull.empty() && !save_vector(w_skull, options.output_w_skull))
        return EXIT_FAILURE;

    if (!options.output_w_fstt.empty() && !save_vector(w_fstt, options.output_w_fstt))
        return EXIT_FAILURE;

    write_report(options.report, options, best, w_skull, w_fstt);

    std::cout << "Final loss: " << best.loss << std::endl;
    std::cout << "Final RMS distance: " << best.rms_distance << std::endl;
    std::cout << "Wrote fitted skin mesh: " << options.output_skin << std::endl;
    std::cout << "Wrote estimated skull mesh: " << options.output_skull << std::endl;
    return EXIT_SUCCESS;
}

//=============================================================================
