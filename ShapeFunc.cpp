
#include "ShapeFunc.h"
#include <vector>
#include <stdexcept>

namespace fem {

ShapeData shapeFunc(double s, double t, int nnode) {
    ShapeData N;
    const int ndime=2;
    N.Shape.assign(nnode , 0.0);//shape functions in natural coordinates [N1 N2 N3 N4]

    //Derivatives of shape functions in natural coordinates
    N.Deriv.assign(ndime,std::vector<double>(nnode , 0.0)); 

    //Tri3 elements
    if (nnode == 3) {
        double p = 1-s-t;

        N.Shape[0] =p; N.Shape[1] =s; N.Shape[2] =t;
        N.Deriv[0][0] = -1; N.Deriv[0][1] = 1; N.Deriv[0][2] = 0;
        N.Deriv[1][0] = -1; N.Deriv[1][1] = 0; N.Deriv[1][2] = 1;
    return N;
    }

    //Quad4 elements 
    if (nnode == 4) {
        double st = s*t;

        N.Shape[0] =(1 - t - s + st)*0.25;
        N.Shape[1] =(1 - t + s - st)*0.25;
        N.Shape[2] =(1 + t + s + st)*0.25;
        N.Shape[3] =(1 + t - s - st)*0.25;

        N.Deriv[0][0] = (-1 + t)*0.25;
        N.Deriv[0][1] = ( 1 - t)*0.25;
        N.Deriv[0][2] = ( 1 + t)*0.25;
        N.Deriv[0][3] = (-1 - t)*0.25;

        N.Deriv[1][0] = (-1 + s)*0.25;
        N.Deriv[1][1] = (-1 - s)*0.25;
        N.Deriv[1][2] = ( 1 + s)*0.25;
        N.Deriv[1][3] = ( 1 - s)*0.25;
        return N;
    }

    //Quad8 elements
    if (nnode == 8) {
        double st = s*t, ss = s*s, tt= t*t;

        N.Shape[0] =(-1 + st + ss + tt - ss*t - s*tt)*0.25;
        N.Shape[1] =( 1 - t - ss + ss*t)*0.5;
        N.Shape[2] =(-1 - st + ss + tt - ss*t + s*tt)*0.25;
        N.Shape[3] =( 1 + s - tt - s*tt)*0.5;
        N.Shape[4] =(-1 + st + ss + tt + ss*t + s*tt)*0.25;
        N.Shape[5] =( 1 + t - ss - ss*t)*0.5;
        N.Shape[6] =(-1 - st + ss + tt + ss*t - s*tt)*0.25;
        N.Shape[7] =( 1 - s - tt + s*tt)*0.5;


        N.Deriv[0][0] = (t + 2*s - 2*s*t - tt)*0.25;
        N.Deriv[0][1] = -s + s*t;
        N.Deriv[0][2] = (-t + 2*s - 2*s*t + tt)*0.25;
        N.Deriv[0][3] = (1 - tt)*0.5;
        N.Deriv[0][4] = (t + 2*s + 2*s*t + tt)*0.25;
        N.Deriv[0][5] = -s - s*t;
        N.Deriv[0][6] = (-t + 2*s + 2*s*t - tt)*0.25;
        N.Deriv[0][7] = (-1 + tt)*0.5;

        N.Deriv[1][0] = (s + 2*t - ss - 2*s*t)*0.25;
        N.Deriv[1][1] = (-1 + ss)*0.5;
        N.Deriv[1][2] = (-s + 2*t - ss + 2*s*t)*0.25;
        N.Deriv[1][3] = -t - s*t;
        N.Deriv[1][4] = (s + 2*t + ss + 2*s*t)*0.25;
        N.Deriv[1][5] = (1 - ss)*0.5;
        N.Deriv[1][6] = (-s + 2*t + ss - 2*s*t)*0.25;
        N.Deriv[1][7] = -t + s*t;
        return N;
    }
    throw std::runtime_error("Unsupported nnode (use 3, 4, 8).");
}

}  // namespace fem