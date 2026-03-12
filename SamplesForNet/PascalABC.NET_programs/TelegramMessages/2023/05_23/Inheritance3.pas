uses GraphWPF;

type
  Shape = auto class
    x,y: real;
  end;
  Circle = class(Shape)
    r: real;
    constructor(x,y,rr: real);
    begin
      inherited Create(x,y);
      r := rr;
    end;
    procedure Draw := GraphWPF.Circle(x,y,r);
    procedure SetRadius(rr: real) := r := rr;
  end;
  Ellipse = class(Shape)
    r,r2: real;
    constructor(x,y,rr,rr2: real);
    begin
      inherited Create(x,y);
      r := rr;
      r2 := rr2;
    end;
    procedure Draw := GraphWPF.Ellipse(x,y,r,r2);
    procedure SetRadius(rr: real) := r := rr;
    procedure SetRadius2(rr2: real) := r2 := rr2;
  end;
  
begin
  var c := new Circle(200,200,100);
  c.Draw;
  var ell := new Ellipse(400,200,50,100);
  ell.Draw;
  c.SetRadius(50);
  c.Draw;
end.  