##
function ToDictionary<T>(Self: sequence of T): Dictionary<T,integer>; extensionmethod
  := Self.ToDictionary(x->x,x->0);

var d := Dict(('a'..'z').ZipTuple(|0|*26));
d.Println;
var d1 := ('a'..'z').ToDictionary;
d1.Println;

