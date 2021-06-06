#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include <math.h>

using namespace arma;
namespace py = pybind11;
class VAR
{
private:
    mat _data;
    int max_lags;
    int n, k;
    const std::vector<std::string> valid_ic{"aic", "fpe", "hqic", "bic"};
    const std::vector<std::string> valid_trend{"c", "ct", "ctt", "nc"};

    /* data */
public:
    VAR(const std::vector<std::vector<double>> &z){
        std::vector<double> z_flat;
        size_t m_nrow = z.size();
        size_t m_ncol = z[0].size();
        for (size_t i = 0; i < m_ncol; ++i) {
            for (size_t j = 0; j < m_nrow; ++j) {
                z_flat.push_back(z[j][i]);
            }
        }
        mat tmp(&z_flat.front(), m_nrow, m_ncol);       
        _data= tmp;
        k = _data.n_cols;
        n = _data.n_rows;
    };
    ~VAR() = default;


    mat get_var_endog(int lags){
        mat Z(n-lags, k*lags);
        int _k = 0;
        for (int t = lags; t < n; t++){
            rowvec a = vectorise(reverse(_data.rows(t-lags, t-1)),1);
            Z.row(_k++) = a;
        }
        return Z;
    }


    void fit(int lags){
        max_lags = int(12 * pow(0.01 * n, 0.25));
        mat tmp = get_var_endog(lags);
    }

    mat _estimate_var(int p){
        mat xt(n-p , (k*p)+1 );
        xt.ones();
        int _k = 0;
        for (int i = 0 ; i < (n-p) ; i++){
            rowvec a(1+k*p);
            a.zeros();
            _k=0;
            a.at(_k++) = 1.0;
            for (int j = 0 ; j < p ; j++){
                a.at(_k++) = _data(i+p-j-1,0);
                a.at(_k++) = _data(i+p-j-1,1);
            }
            xt.row(i) = a;
        }
        mat zt = _data.rows( p, n-1 );
        mat beta = ( xt.t() * xt ).i() * xt.t() * zt ;
        
        mat A = zt - (xt * beta);
        mat sigma = ( (A.t()) * A ) / (double)(n-p);  
        return sigma;
    }

    uword order_select(int max_p){
        rowvec bic(max_p);
        bic.zeros();
        for (int p = 1; p < max_p + 1; p++){
            mat sigma = _estimate_var(p);
            bic.at(p-1) =  log(det(sigma)) + log(n) * (double)p * (double)(k*k) / (double)n;
        }
        uword bic_order = bic.index_min() + 1;
        
        return bic_order;
    }

};

PYBIND11_MODULE(VAR, m) {
    m.doc() = "VAR model implementation";
    py::class_<VAR>(m, "VAR", py::buffer_protocol())
        .def(py::init<std::vector<std::vector<double>>&>())
        .def("fit", &VAR::fit)
        .def("order_select", &VAR::order_select);
}