type
  ArrComparer = class(IEqualityComparer<array of integer>)
    public function Equals(a,b: array of integer) := a.ArrEqual(b);
    function GetHashCode(a: array of integer) := 1;
  end;

var aa := ||1,2|,|2,3|,|1,2|,|6,4|,|1,2|,|2,3|,|5,5||;

begin
  aa.Println;
  aa.Distinct(new ArrComparer).Println
end.

