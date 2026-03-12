uses GraphWPF;

type
  Ellipse = class
    x,y,r,r2: real;
    constructor(xx,yy,rr,rr2: real) := (x,y,r,r2) := (xx,yy,rr,rr2);
    procedure Draw := GraphWPF.Ellipse(x,y,r,r2);
    procedure SetRadius(rr: real) := r := rr;
    procedure SetRadius2(rr2: real) := r2 := rr2;
  end;
  Circle = class(Ellipse)    
    constructor(x,y,r: real) := inherited Create(x,y,r,r);
    procedure Draw := GraphWPF.Circle(x,y,r);
    procedure SetRadius(rr: real) := (r,r2) := (rr,rr);
    procedure SetRadius2(rr2: real) := raise new System.NotSupportedException;
  end;
  
begin
  var c := new Circle(200,200,100);
  c.Draw;
  var ell := new Ellipse(400,200,50,100);
  ell.Draw;
  c.SetRadius(50);
  c.Draw;
end.  