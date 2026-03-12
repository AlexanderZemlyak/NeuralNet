type A = class(IComparer<integer>)
public
  function Compare(a,b: integer) := a <= b ? -1 : 1;
end;

begin
  var s := new SortedSet<integer>(Arr(1,2,3,3,2),new A);
  s.Print;
end.