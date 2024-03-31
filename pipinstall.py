python --version
Python 3.N.N
python -m pip --version
pip X.Y.Z from ... (python 3.N.N)

C:> py -m pip install sampleproject
[...]
Successfully installed sampleproject
C:> py -m pip install sampleproject-1.0.tar.gz
[...]
Successfully installed sampleproject
C:> py -m pip install sampleproject-1.0-py3-none-any.whl
[...]
Successfully installed sampleproject
py -m venv tutorial_env
tutorial_env\Scripts\activate
#Euler
from math import *
def euler(f, x0, y0, h, n):
    x_values = [x0]
    y_values = [y0]
    for i in range(1, n+1):
        x_new = x_values[i - 1] + h
        y_new = y_values[i - 1] + h * f(x_values[i - 1], y_values[i - 1])
        x_values.append(x_new)
        y_values.append(y_new)
    return x_values, y_values
equation_input = input("Enter the differential equation in terms of x and y: ")
equation = lambda x, y: eval(equation_input)
x0 = float(input("Enter initial value of x: "))
y0 = float(input("Enter initial value of y: "))
h = float(input("Enter step size (h): "))
n = int(input("Enter number of iterations (n): "))
x_values, y_values = euler(equation, x0, y0, h, n)
print("\nResults: ")
for i in range(len(x_values)):
    print(f"Iteration {i }: x = {x_values[i]}, y = {y_values[i]}")


#NEWTON
from sympy import sympify, diff
def newton_raphson(f, x, iterations):
    f = sympify(f)
    var, = f.free_symbols
    df = diff(f)
    print(f'itr|    {var}')
    for i in range(1, iterations+1):
        fx = f.subs(var, x)
        dfx = df.subs(var, x)
        x = float(x-(fx/dfx))
        print(f'{i:02} | {x:.8f}')
    return x
if __name__=='__main__':
    f = input('Enter the function: ')
    x0 = float(input('Enter the initial guess: '))
    itr = int(input('Enter the number of iterations: '))
    print(newton_raphson(f, x0, itr))

#RK4
from math import *
def runge_kutta_4th_order(f, x0, y0, h, n):
    x_values = [x0]
    y_values = [y0]
    for i in range(1, n+1):
        k1 = h * f(x_values[i - 1], y_values[i - 1])
        k2 = h * f(x_values[i - 1] + (h/2), y_values[i - 1] + (k1/2))
        k3 = h * f(x_values[i - 1] + (h/2), y_values[i - 1] + (k2/2))
        k4 = h * f(x_values[i - 1] + h, y_values[i - 1] + k3)
        y_new = y_values[i - 1] +  ((1/6)* (k1 + 2*k2 + 2*k3 + k4))
        x_new = x_values[i - 1] + h
        x_values.append(x_new)
        y_values.append(y_new)
    return x_values, y_values
equation_input = input("Enter the differential equation in terms of x and y: ")
equation = lambda x, y: eval(equation_input)
x0 = float(input("Enter initial value of x: "))
y0 = float(input("Enter initial value of y: "))
h = float(input("Enter step size (h): "))
n = int(input("Enter number of iterations (n): "))
x_values, y_values = runge_kutta_4th_order(equation, x0, y0, h, n)
print("\nResults: ")
for i in range(len(x_values)):
    print(f"Iteration {i }: x = {x_values[i]}, y = {y_values[i]}")

/*py -m venv tutorial_env
tutorial_env\Scripts\activate */

#RK2
from math import *
def runge_kutta_2nd_order(f, x0, y0, h, n):
    x_values = [x0]
    y_values = [y0]

    for i in range(1, n+1):
        k1 = h * f(x_values[i - 1], y_values[i - 1])
        k2 = h * f(x_values[i - 1] + h, y_values[i - 1] + k1)
        y_new = y_values[i - 1] + 0.5 * (k1 + k2)
        x_new = x_values[i - 1] + h

        x_values.append(x_new)
        y_values.append(y_new)

    return x_values, y_values

equation_input = input("Enter the differential equation in terms of x and y: ")
equation = lambda x, y: eval(equation_input)

x0 = float(input("Enter initial value of x: "))
y0 = float(input("Enter initial value of y: "))
h = float(input("Enter step size (h): "))
n = int(input("Enter number of iterations (n): "))

x_values, y_values = runge_kutta_2nd_order(equation, x0, y0, h, n)

print("\nResults: ")
for i in range(len(x_values)):
    print(f"Iteration {i }: x = {x_values[i]}, y = {y_values[i]}")

#HEUN
from math import *
def huen(f, x0, y0, h, n):
    x_values = [x0]
    y_values = [y0]

    for i in range(1, n+1):
        x_new = x_values[i - 1] + h
        y_new = y_values[i - 1]+ (h/2)*(f(x_values[i - 1], y_values[i - 1]) + f(x_new, (y_values[i - 1] + h * f(x_values[i - 1], y_values[i - 1]))))

        x_values.append(x_new)
        y_values.append(y_new)

    return x_values, y_values

equation_input = input("Enter the differential equation in terms of x and y: ")
equation = lambda x, y: eval(equation_input)

x0 = float(input("Enter initial value of x: "))
y0 = float(input("Enter initial value of y: "))
h = float(input("Enter step size (h): "))
n = int(input("Enter number of iterations (n): "))

x_values, y_values = huen(equation, x0, y0, h, n)

print("\nResults: ")
for i in range(len(x_values)):
    print(f"Iteration {i }: x = {x_values[i]}, y = {y_values[i]}")

#TRAP
def f(eq,x):
    eq=eq.replace("x",str(x))
    return float (eval(eq))
def trap(lower, upper, iteration, eq):
    h = float((upper - lower)/iteration)
    result = f(eq, upper) +f(eq, lower)
    for i in range(1, iteration):
        x = lower + i *h
        result += 2*f(eq,x)
    return((float(h)/2)*(result))
eq= input("Enter Equation:")
u = float(input("Enter upper limit: "))
l= float(input("Enter lower limit: "))
n = int(input("How many iterations are required: "))
result = trap(l,u,n,eq)
print("Answer using trapezoidal rule : ",result)

#SIM3
def func(eq, x):
    eq = eq.replace("x", str(x))
    return float(eval(eq))
def simp3(l, u, n, eq):
    h = float((u - l) / n)
    res = func(eq, l) + func(eq, u)
    x = l
    for i in range(1, n):
        x += h
        if(i % 2 != 0):
            res += 4 * func(eq, x)
        else:
            res += 2 * func(eq, x)
    return str(h / 3 * res)
if __name__ == "__main__":
    eq = input("Enter the equation: ")
    l = float(input("Enter the lower limit: "))
    u = float(input("Enter the upper limit: "))
    n = int(input("Enter the number of iterations: ")) 
    print("Answer for Simpson's 1/3rd rule: ", simp3(l, u, n, eq))

#SIM3-8
def func(eq,x):
    eq = eq.replace("x", str(x))
    return float(eval(eq))
def simpson_3_8(l, u, n, eq):
    h = float((u - l) / n)
    res = func(eq, l) + func(eq, u)
    x = l
    for i in range(1, n):
        x += h
        if i % 3 != 0:
            res += 3 * func(eq, x)
        else:
            res += 2 * func(eq, x)
    return str((3 * h / 8) * res)
if __name__ == "__main__":
    eq = input("Enter the equation: ")
    l = float(input("Enter the lower limit: "))
    u = float(input("Enter the upper limit: "))
    n = int(input("Enter the number of intervals (must be a multiple of 3): "))
    if n % 3 != 0:
        print("Number of intervals must be a multiple of 3 for Simpson's 3/8 rule.")
    else:
        print("Answer for Simpson's 3/8 rule:", simpson_3_8(l, u, n, eq))


#NEWTON FARWARD 
from math import factorial
def fnewton_interpolation(x,X,Y):
    h=X[1]-X[0]
    u=(x-X[0])/h
    fx=Y[0]
    perm=1
    for i in range (len(X)-1):
        dy=[]
        for j in range (len(Y)-1):
            dy.append(Y[j+1]-Y[j])
        perm*=(u-i)
        fx+=(perm*dy[0])/factorial(i+1)
        y=dy
    return fx
if __name__ =="__main__":
    x=float(input("Enter the value of x: "))
    X=list(map(float, input("Enter the value of X(comma-separated): ").split(',')))
    Y=list(map(float, input("Enter the value of Y(comma-separated): ").split(',')))
    print("Interpolation value :",fnewton_interpolation(x,X,Y))

#NEWTON bACKWARD
from math import factorial
def bnewton_interpolation(x,X,Y):
    h=X[1]-X[0]
    u=(x-X[-1])/h
    fx=Y[-1]
    perm=1
    for i in range (len(X)-1):
        dy=[]
        for j in range (len(Y)-1):
            dy.append(Y[j+1]-Y[j])
        perm*=(u+i)
        fx+=(perm*dy[-1])/factorial(i+1)
        y=dy
    return fx
if __name__ =="__main__":
    x=float(input("Enter the value of x: "))
    X=list(map(float, input("Enter the value of X(comma-separated): ").split(',')))
    Y=list(map(float, input("Enter the value of Y(comma-separated): ").split(',')))
    print("Interpolation value :",bnewton_interpolation(x,X,Y))

#LAGNG
from math import prod
def lagrange_interpolation(x,X,Y):
    fx=0
    for i in range(len(X)):
        num = prod(x-X[j] for j in range  (len(X)) if j!=i)
        denom = prod(X[i]-X[j] for j in range (len(X)) if j!=i)
        fx += Y[i]*num/denom
    return fx
if __name__ =="__main__":
    x=float(input("Enter the value of x: "))
    X=list(map(float, input("Enter the value of X(comma-separated): ").split(',')))
    Y=list(map(float, input("Enter the value of Y(comma-separated): ").split(',')))
    print("Interpolation value :",lagrange_interpolation(x,X,Y))
    


#REGULA FALSI
from sympy import sympify
def Regula_false(f, a, b, iterations):
        f = sympify(f)
        var, = f.free_symbols
        fa = f.subs(var, a)
        fb = f.subs(var, b)
        if fa*fb>=0:
            raise ValueError(f'The root does not lie in the open interval({a}, {b})')
        print(f'itr|   a  |    b    | {var}')
        for i in range(1, iterations+1):
            x = (a*fb-b*fa)/(fb-fa)
            fx = f.subs(var, x)
            print(f'{i:02} |  {a:.8f}  | {x:.8f}')
            if fx == 0:
                print('Root found: further iterations not possible')
                break
            elif fx*fa<0:
                b = x
                fb = fx
            else:
                a = x
                fa = fx
        return x
if __name__ =='__main__':
    f = input('Enter the function: ')
    a = float(input('Enter the lower limit: '))
    b = float(input('Enter the upper limit: '))
    itr = int(input('Enter the number of iterations: '))
    print(Regula_false(f, a, b, itr))

#SECANT
from sympy import sympify

def secant(f, x0, x1, itr):
    f = sympify(f)
    var, = f.free_symbols
    fx0 = f.subs(var, x0)
    fx1 = f.subs(var, x1)
    
    print(f'itr | x0 | x1 | {var}')
    for i in range(1, itr + 1):
        x = x1 - (fx1 * (x1 - x0)) / (fx1 - fx0)
        fx = f.subs(var, x)
        print(f'{i:02} | {x0:.4f} | {x1:.4f} | {x:.4f}')
        
        if fx == 0:
            print('Root found: further iteration not possible')
            return x
        x0, x1 = x1, x
        fx0, fx1 = fx1, fx
    return x

if __name__ == '__main__':
    f = input("Enter a Function: ")
    x0 = float(input("Enter the first initial guess (x0): "))
    x1 = float(input("Enter the second initial guess (x1): "))
    itr = int(input("Enter number of iterations: "))
print(secant(f, x0, x1, itr))

#BISECTION
from sympy import sympify
def bisection(f, a, b, iterations):
        f = sympify(f)
        var, = f.free_symbols
        fa = f.subs(var, a)
        fb = f.subs(var, b)
        if fa*fb>=0:
            raise ValueError(f'The root does not lie in the open interval({a}, {b})')
        print(f'itr|   a  |    b    | {var}')
        for i in range(1, iterations+1):
            x = (a+b)/2
            fx = f.subs(var, x)
            print(f'{i:02} |  {a:.8f}  | {x:.8f}')
            if fx == 0:
                print('Root found: further iterations not possible')
                break
            elif fx*fa<0:
                b = x
                fb = fx
            else:
                a = x
                fa = fx
        return x
if __name__ =='__main__':
    f = input('Enter the function: ')
    a = float(input('Enter the lower limit: '))
    b = float(input('Enter the upper limit: '))
    itr = int(input('Enter the number of iterations: '))
    print(bisection(f, a, b, itr))

#CENTRAL FORWARD
from math import factorial

def GUSS_CENTRAL(x, X, Y):
    h = X[1] - X[0]
    u = (x - X[0]) / h
    fx = Y[0]
    perm = 1
    for i in range(len(X) - 1):
        dy = []
        for j in range(len(Y) - i - 1):
            dy.append(Y[j + 1] - Y[j])
        perm *= (u - i / 2)
        fx += (perm * dy[0]) / factorial(i + 1)
        Y = dy
    return fx

if __name__ == "__main__":
    x = float(input("Enter the value of x: "))
    X = list(map(float, input("Enter the value of X (comma-separated): ").split(',')))
    Y = list(map(float, input("Enter the value of Y (comma-separated): ").split(',')))
    print("Interpolation value:", GUSS_CENTRAL(x, X, Y))

#GAUSS SEIDEL
def gauss_seidel(mat, iterations, initial_vec=None):
    f=[]
    for i, row in enumerate(mat):
        var=-row.pop(i)
        row[-1]*=-1
        eq=[elem/var for elem in row]
        f.append(eq)
    x=[0]*len(f) if initial_vec is None else initial_vec
    for k in range(iterations):
        for i in range(len(f)):
            rest=x+[1]
            rest.pop(i)
            x[i]=sum(coeff*var for coeff, var in zip(f[i], rest))
        print(k+1,x)
    return x
if __name__ == '__main__':
    M=[[20,1,-2,17],[3,20,-1,-18],[2,-3,20,25]]
    print("Matrix M:\n",M)
    itr=int(input("ENter the no. of iterations:"))
    print("Solution M:",gauss_seidel(M,itr))

#JACOBI
def jacobi(mat, iterations, initial_vec=None):
    f = []
    for i, row in enumerate(mat):
        var = -row.pop(i)
        row[-1] *= -1
        eq = [elem / var for elem in row]
        f.append(eq)
    x = [0] * len(f) if initial_vec is None else initial_vec
    for k in range(iterations):
        x_new = [0] * len(x)
        for i in range(len(f)):
            rest = x[:]
            rest.pop(i)
            x_new[i] = sum(coeff * var for coeff, var in zip(f[i], rest))
        x = x_new
        print(k + 1, x)
    return x
if __name__ == '__main__':
    M = [[20, 1, -2, 17], [3, 20, -1, -18], [2, -3, 20, 25]]
    print("Matrix M:\n", M)
    itr = int(input("Enter the number of iterations: "))
    print("Solution M:", jacobi(M, itr))


                

            






