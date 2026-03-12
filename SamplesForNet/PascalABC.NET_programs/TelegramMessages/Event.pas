##
uses GraphWPF, Controls;

var b := Button(10,10,'Случайный круг');
b.Click := () -> Circle(RandomPoint(100),30,RandomColor);