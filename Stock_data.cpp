#include <fstream>
#include "Stock_data.hpp"
#include <string.h>
#include <sstream>
#include <set>
using namespace std;

void Stock_data::read_csv(string _filename){

    ifstream myFile(_filename);

    if(!myFile.is_open()) throw runtime_error("Could not open file");
    string line, colname;
    float val;

    // Read the column names
    if(myFile.good())
    {
        // Extract the first line in the file
        getline(myFile, line);

        // Create a stringstream from line
        stringstream ss(line);

        // Extract each column name
        int i = 0;
        while(getline(ss, colname, ',')){
            _company.insert(make_pair(colname,i));
            _data.push_back(vector<float>());
        }
    }
    // Read data, line by line
    while(std::getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);
        
        // Keep track of the current column index
        int colIdx = 0;
        
        // Extract each integer
        while(ss >> val){
            
            // Add the current integer to the 'colIdx' column's values vector
            _data.at(colIdx).push_back(val);
            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();
            
            // Increment the column index
            colIdx++;
        }
    }
}

vector<float> Stock_data::operator() (string _id) const{
    if (_company.find(_id) == _company.end())
        throw runtime_error("key error : comapny not exists. ");
    return _data[_company.at(_id)];
}
// ostream & operator<<(ostream &output, const Stock_data &d){
//     for (std::vector<int>::iterator it = d._data.begin() ; it != myvector.end(); ++it)
//         std::cout << ' ' << *it;
// }