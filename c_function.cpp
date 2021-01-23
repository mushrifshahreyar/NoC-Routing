#include<bits/stdc++.h>
#include <cstdio>
#include <string>
#include <Python.h>
#include "pyhelper.hpp"

using namespace std;

int main() {
    
    setenv("PYTHONPATH", ".", 1);
	CPyInstance hInstance;

    CPyObject pName = PyUnicode_FromString("test");
	CPyObject pModule = PyImport_Import(pName);
	
	if(pModule)
	{   
		CPyObject pFunc = PyObject_GetAttrString(pModule, "getInteger");
		if(pFunc && PyCallable_Check(pFunc))
		{
            CPyObject args = PyTuple_Pack(4,PyFloat_FromDouble(2.0),PyFloat_FromDouble(4.0),PyFloat_FromDouble(6.0),PyFloat_FromDouble(8.0));
			CPyObject pValue = PyObject_CallObject(pFunc, args);

			printf("C: getInteger() = %ld\n", PyLong_AsLong(pValue));
		}
		else
		{
			printf("ERROR: function getInteger()\n");
		}

	}
	else
	{
		printf("ERROR: Module not imported\n");
	}

	return 0;
    return 0;
}
