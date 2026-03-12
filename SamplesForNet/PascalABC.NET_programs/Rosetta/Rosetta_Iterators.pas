// https://rosettacode.org/wiki/Iterators#PascalABC.NET

begin
  var a: array of string := Arr('Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday');
  var L: LinkedList<string> := LLst('Red','Orange','Yellow','Green','Blue','Purple');
  a.Println;
  L.Println;
  
  var it := a.GetEnumerator;
  it.MoveNext;
  Print(it.Current);
  it.MoveNext; it.MoveNext; it.MoveNext;
  Print(it.Current);
  it.MoveNext;
  Print(it.Current);
end.