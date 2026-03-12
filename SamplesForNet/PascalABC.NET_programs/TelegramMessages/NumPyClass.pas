type 
  MyArr<T> = class(IEnumerable<T>)
  private
    a: array of T;
  public
    function GetEnumerator(): IEnumerator<T> := (a as IEnumerable<T>).GetEnumerator();
    function System.Collections.IEnumerable.GetEnumerator: System.Collections.IEnumerator := Self.GetEnumerator;
    constructor(aa: array of T) := a := aa;
    function ToString: string; override := '['+a.JoinToString(',')+']';
    static function operator implicit(a: sequence of T): MyArr<T> := new MyArr<T>(a.ToArray);
  end;
  NumPabc = static class
  public
    static function &Array<T>(aa: array of T): MyArr<T> := new MyArr<T>(aa);
    static function &Array<T>(aa: sequence of T): MyArr<T> := new MyArr<T>(aa.ToArray);
  end;
  np = NumPabc;
  
function Sin(a: MyArr<real>): MyArr<real> := a.Select(PABCSystem.Sin);
function Cos(a: MyArr<real>): MyArr<real> := a.Select(PABCSystem.Cos);
function Ln(a: MyArr<real>): MyArr<real> := a.Select(PABCSystem.Ln);
function Round(a: MyArr<real>): MyArr<integer> := a.Select(x->PABCSystem.Round(x));
function Round(a: MyArr<real>; n: integer): MyArr<real> := a.Select(x->PABCSystem.Round(x,n));
function All(a: MyArr<boolean>): boolean := a.All(x->x);
function Any(a: MyArr<boolean>): boolean := a.Any(x->x);

function operator>(a,b: MyArr<real>); extensionmethod := a.Zip(b,(x,y)->x>y);
function operator<(a,b: MyArr<real>); extensionmethod := a.Zip(b,(x,y)->x<y);
function operator>=(a,b: MyArr<real>); extensionmethod := a.Zip(b,(x,y)->x>=y);
function operator<=(a,b: MyArr<real>); extensionmethod := a.Zip(b,(x,y)->x<=y);
function operator=(a,b: MyArr<real>); extensionmethod := a.Zip(b,(x,y)->x=y);
function operator<>(a,b: MyArr<real>); extensionmethod := a.Zip(b,(x,y)->x<>y);
function operator+(a,b: MyArr<real>); extensionmethod := a.Zip(b,(x,y)->x+y);
function operator-(a,b: MyArr<real>); extensionmethod := a.Zip(b,(x,y)->x-y);
function operator*(a,b: MyArr<real>); extensionmethod := a.Zip(b,(x,y)->x*y);
function operator/(a,b: MyArr<real>); extensionmethod := a.Zip(b,(x,y)->x/y);
function operator**(a,b: MyArr<real>); extensionmethod := a.Zip(b,(x,y)->x**y);

function operator>(a: MyArr<real>; x: real); extensionmethod := a.Select(y -> y>x);
function operator<(a: MyArr<real>; x: real); extensionmethod := a.Select(y -> y<x);
function operator>=(a: MyArr<real>; x: real); extensionmethod := a.Select(y -> y>=x);
function operator<=(a: MyArr<real>; x: real); extensionmethod := a.Select(y -> y<=x);
function operator=(a: MyArr<real>; x: real); extensionmethod := a.Select(y -> y=x);
function operator<>(a: MyArr<real>; x: real); extensionmethod := a.Select(y -> y<>x);

function operator+(a: MyArr<real>; x: real); extensionmethod := a.Select(y -> y+x);
function operator+(x: real; a: MyArr<real>); extensionmethod := a.Select(y -> y+x);
function operator-(a: MyArr<real>; x: real); extensionmethod := a.Select(y -> y-x);
function operator-(x: real; a: MyArr<real>); extensionmethod := a.Select(y -> x-y);
function operator*(a: MyArr<real>; x: real); extensionmethod := a.Select(y -> y*x);
function operator*(x: real; a: MyArr<real>); extensionmethod := a.Select(y -> y*x);
function operator/(a: MyArr<real>; x: real); extensionmethod := a.Select(y -> y/x);
function operator/(x: real; a: MyArr<real>); extensionmethod := a.Select(y -> x/y);
function operator**(a: MyArr<real>; x: real); extensionmethod := a.Select(y -> y ** x);
function operator**(x: real; a: MyArr<real>); extensionmethod := a.Select(y -> x ** y);
function operator**(x: integer; a: MyArr<real>); extensionmethod := a.Select(y -> x ** y);

begin
  var x := np.Array(|1.0,2.0,6.0|);
  var y := np.Array(|5.0,4.0,3.0|);
  Println(x > y);
  Println(x * y);
  Println(x + 2.5*y + 1.5);
  Println(x / y);
  Println(x ** y);
  Println(x ** 2);
  Println(2 ** x);
  Println(Round(Sin(x),2));
  Println(Round(Cos(y),2));
  Println(All(x>1));
end.