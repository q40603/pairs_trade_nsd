#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
using namespace std;
class Stock_data
{
public:
    Stock_data() = default;
    ~Stock_data() = default;
    void read_csv(std::string _file);
    void drop_empty();
    void to_log();
    void drop_stationary();
    vector<string> get_company_list();
    vector<double> operator() (string _id) const;
    friend ostream& operator<<(ostream &print, const Stock_data &d);
private:
   unordered_map<string, vector<double> > _data;
};