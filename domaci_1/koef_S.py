if __name__ == '__main__':
    num = 3140
    suma = 0
    while num > 0:
        suma += num % 10
        num //= 10

    print('S = {:d}\nS mod 3 = {:d}'.format(suma, suma % 3))
