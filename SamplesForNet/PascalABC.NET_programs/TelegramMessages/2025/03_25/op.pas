##
var operations := new Dictionary<string, function(x, y: real): real>;
operations['+'] := (x, y) -> x + y;
operations['-'] := (x, y) -> x - y;

var res := operations['+'](2, 3); 
res.Println;
res := operations['-'](2, 3); 
res.Println;