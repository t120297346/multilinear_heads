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
#pragma once
//=============================================================================

#include "MultilinearModel.h"

#include <pmp/SurfaceMesh.h>
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <iostream>


//== HELPER ===================================================================

//! read matrix from binary file
static bool load_matrix(Eigen::MatrixXd& eigenMatrix,
                        const std::string& filename)
{
    std::ifstream ifs( filename.c_str(), std::ofstream::binary );
    if (!ifs.is_open())
    {
        std::cerr << "Cannot load matrix " << filename << std::endl;
        return false;
    }
    unsigned int n_rows = 0;
    unsigned int n_cols = 0;
    ifs.read(reinterpret_cast<char *>(&n_rows), sizeof(n_rows));
    ifs.read(reinterpret_cast<char *>(&n_cols), sizeof(n_cols));
    eigenMatrix.resize(n_rows, n_cols);
    ifs.read(reinterpret_cast<char *>(eigenMatrix.data()), n_rows*n_cols*sizeof(typename Eigen::MatrixXd::Scalar) );
    ifs.close();

    return true;
}

//-----------------------------------------------------------------------------

//! read vector from binary file
static bool load_vector(Eigen::VectorXd& eigenVector,
                        const std::string& filename)
{
    std::ifstream ifs( filename.c_str(), std::ofstream::binary );
    if (!ifs.is_open())
    {
        std::cerr << "Cannot load vector " << filename << std::endl;
        return false;
    }
    unsigned int rows = 0;
    ifs.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    eigenVector = Eigen::VectorXd(rows);
    ifs.read(reinterpret_cast<char *>(eigenVector.data()), rows*sizeof(typename Eigen::VectorXd::Scalar) );
    ifs.close();

    return true;
}

//-----------------------------------------------------------------------------

//! read vector with scalars from text file
static bool load_scalars(std::vector<double>& output, const std::string& filename)
{
    std::ifstream ifs(filename);
    if(!ifs.is_open())
    {
        std::cerr << "Cannot read weights from " << filename << std::endl;
        return false;
    }

    output.clear();

    double val;
    while (true)
    {
        ifs >> val;
        if (ifs.eof()) break;
        output.push_back(val);
    }

    ifs.close();

    return (!output.empty());
}

//-----------------------------------------------------------------------------

//! load parameters for skull shape and FSTT distribution from text files
static bool load_parameters(Eigen::VectorXd& wSkull,
                            Eigen::VectorXd& wFstt,
                            const std::string& filenamewSkull, 
                            const std::string& filenamewFstt)
{
    std::vector<double> vecwSkull;
    if (!load_scalars(vecwSkull, filenamewSkull))
    {
        std::cerr << "[ERROR] in 'load_parameters(...)' - Can't load w_skull" << std::endl;
        return false;
    }

    std::vector<double> vecwFstt;
    if (!load_scalars(vecwFstt, filenamewFstt))
    {
        std::cerr << "[ERROR] in 'load_parameters(...)' - Can't load w_fstt" << std::endl;
        return false;
    }

    if ((int)wSkull.size() > (int)vecwSkull.size() ||
        (int)wFstt.size()  > (int)vecwFstt.size())
    {
        std::cerr << "[ERROR] in 'load_parameters(...)' - Loaded w_skull and w_fstt are too short" << std::endl;
        return false;
    }
    if (wSkull.size() == 0 || wFstt.size() == 0)
    {
        std::cerr << "[ERROR] in 'load_parameters(...)' - No previously loaded parameters w_skull and w_fstt available" << std::endl;
        return false;
    }

    for (unsigned int i = 0; i < wSkull.size(); ++i)
    {
        wSkull(i) = vecwSkull[i];
    }

    for (unsigned int i = 0; i < wFstt.size(); ++i)
    {
        wFstt(i) = vecwFstt[i];
    }

    return true;
}

//=============================================================================

//! ensure directory names can be used as filename prefixes
static std::string with_trailing_separator(const std::string& path)
{
    if (path.empty())
        return path;

    const char last = path[path.size() - 1];
    if (last == '/' || last == '\\')
        return path;

    return path + "/";
}

//-----------------------------------------------------------------------------

//! save an Eigen vector as one scalar per line
static bool save_vector(const Eigen::VectorXd& values,
                        const std::string& filename)
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

//-----------------------------------------------------------------------------

//! initialize model weights with the same mean setting used by MLMViewer
static void initialize_mean_parameters(Eigen::VectorXd& wSkull,
                                       Eigen::VectorXd& wFstt,
                                       const MultilinearModel& model)
{
    wSkull = Eigen::VectorXd::Zero(model.dim1());
    for (int i = 0; i < model.U_skull().rows(); ++i)
    {
        for (int j = 0; j < model.U_skull().cols(); ++j)
            wSkull(j) += model.U_skull()(i, j);
    }
    wSkull /= static_cast<double>(model.U_skull().rows());

    wFstt = Eigen::VectorXd::Zero(model.dim2());
    for (int i = 0; i < model.U_fstt().rows(); ++i)
    {
        for (int j = 0; j < model.U_fstt().cols(); ++j)
            wFstt(j) += model.U_fstt()(i, j);
    }
    wFstt /= static_cast<double>(model.U_fstt().rows());
}

//=============================================================================
