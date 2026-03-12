##
var x := 12345;
// Реализация Unfold - метода развертки для последовательности
var q := x.Iterate(x -> x div 10).Select(x -> x mod 10).TakeWhile(x -> x > 0);
q.Println;
q.Reverse.Aggregate(0,(num,x) -> num * 10 + x).Println;
