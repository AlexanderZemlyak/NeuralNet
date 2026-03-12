##
var n := 1726354;
n.Println;
// Развёртка без проекции
var seq := SeqWhile(n, n -> n div 10, n -> n > 0);
seq.Println;
// Проекция - отдельно
seq.Select(n -> n mod 10).Println;
seq.Select(n -> n mod 10).Reverse.Println;
