#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "matrix.hpp"


namespace py = pybind11;

class VAR
{
private:
    Matrix _data;
    const std::vector<std::string> valid_ic{'aic', 'fpe', 'hqic', 'bic'};
    const std::vector<std::string> valid_trend{"c", "ct", "ctt", "nc"}

    /* data */
public:
    VAR(std::vector<std::vector<double>> const &z){
        _data= Matrix(z);
    };
    ~VAR() = default;

    void fit(std::string method='ols', std::sting ic = "aic", std::string trend = "c"){
        int maxlags = int(round(12*(len(self.endog)/100.)**(1/4.)))
    }

};

PYBIND11_MODULE(VAR, m) {
    m.doc() = "VAR model implementation";
    py::class_<VAR>(m, "VAR", py::buffer_protocol())
        .def(py::init<std::vector<std::vector<double>>&>());
}

// int main(){
//     Matrix _data(0,0);

//     return 0;
// }