function operator*(a: array of real; r: real): array of real; extensionmethod
  := a.ConvertAll(x -> x * r);
  
function operator*(r: real; a: array of real): array of real; extensionmethod := a * r;

function operator*(a,b: array of real): real; extensionmethod
  := a.Zip(b,(xa,xb) -> xa + xb).Sum;

function operator+(a,b: array of real): array of real; extensionmethod
  := a.ConvertAll((x,i) -> x + b[i]);

function operator*(x: array of array of real; w: array of real): array of real; extensionmethod
  := x.ConvertAll(r -> r * w);

function Transp(x: array of array of real): array of array of real
  := MatrByRow(x).Cols;

begin
  var a := MatrRandomReal(3,4, digits := 2).Rows;
  a.Println;
  var b := ArrRandomReal(4);
  b.Println;
  (a * b).Println;
  (Transp(a) *|1.0,2,3|).Println;
end.