uses System.Threading.Tasks;

function Fun1(n: integer): real := (1..n).Sum(i -> Sin(i));
function Fun2(n: integer): real := (1..n).Sum(i -> Cos(i));
function Fun3(n: integer): real := (1..n).Sum(i -> Sin(i*i));

begin
  var n := 30000000;
  MillisecondsDelta;
  Println(Fun1(n)+Fun2(n)+Fun3(n));
  MillisecondsDelta.Println;

  var t1 := Task&<real>.Factory.StartNew(()->Fun1(n));
  var t2 := Task&<real>.Factory.StartNew(()->Fun2(n));
  var t3 := Task&<real>.Factory.StartNew(()->Fun3(n));
  Println(t1.Result+t2.Result+t3.Result);
  MillisecondsDelta.Println;
end.