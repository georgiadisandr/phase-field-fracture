
#ifndef SHAPEFUNC_H
#define SHAPEFUNC_H
#include <vector>

namespace fem {

struct ShapeData {
    std::vector<double> Shape;//Shape Functions
    std::vector<std::vector<double>> Deriv; // Derivatives of shape Functions
};
ShapeData shapeFunc(double s,double t, //natural coordinates
                    int nnode);//number of node per element 

}  // namespace fem

#endif //SHAPEFUNC_H
