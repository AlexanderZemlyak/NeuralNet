uses GraphWPF;

type
  Circle = class
    x,y,r: real;
    constructor(xx,yy,rr: real) := (x,y,r) := (xx,yy,rr);
    procedure Draw := GraphWPF.Circle(x,y,r);
  end;
  Ellipse = class(Circle)
    r2: real;
    constructor(xx,yy,rr,rr2: real);
    begin
      inherited Create(xx,yy,rr);
      r2 := rr2;
    end;  
    procedure Draw := GraphWPF.Ellipse(x,y,r,r2);
  end;
  
begin
  var c := new Circle(200,200,100);
  c.Draw;
  var ell := new Ellipse(400,200,50,100);
  ell.Draw;
end.  