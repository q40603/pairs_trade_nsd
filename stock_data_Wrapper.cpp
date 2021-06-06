#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include "Stock_data.hpp"

namespace py = pybind11;

PYBIND11_MODULE(Stock_data, m) {
    m.doc() = "Stock_data model implementation";
    py::class_<Stock_data>(m, "Stock_data", py::buffer_protocol())
        .def(py::init<>())
        .def("read_csv", &Stock_data::read_csv)
        .def("drop_empty", &Stock_data::drop_empty)
        .def("to_log", &Stock_data::to_log)
        .def("drop_stationary", &Stock_data::drop_stationary)
        .def("get_company_list", &Stock_data::get_company_list)
        .def("keep", &Stock_data::keep)
        .def("__getitem__", [](const Stock_data &st, std::pair<std::string, std::string> i){
            
            vector<double> s1 = st(i.first);
            vector<double> s2 = st(i.second);
            vector<vector <double>> vect_arr;
            vect_arr.push_back(s1);
            vect_arr.push_back(s2);
            py::array ret =  py::cast(vect_arr);
            return ret;
        })
        .def("__getitem__", [](const Stock_data &st, std::string i){
            vector<double> s = st(i);
            return py::array(s.size(), s.data());
        });
}