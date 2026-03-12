uses GraphWPF;

var (m,n) := (30,40);

var a := MatrFill(m,n,0);
var b := Copy(a);
var (x0,y0,w) := (n div 2, m div 2,20);

procedure Draw := 
  foreach var (y,x) in (0..m-1).Cartesian(0..n-1) do
    if a[y,x]=0 then
      Rectangle(x*w,y*w,w-2,w-2)
    else Rectangle(x*w,y*w,w-2,w-2,Colors.Gray); 

procedure Init;
begin
  a[y0,x0] := 1; a[y0,x0+1] := 1; a[y0,x0-1] := 1;
  a[y0+1,x0] := 1; a[y0-1,x0+1] := 1; 
  Draw;
end;

function InhCount(x,y: integer) := a[y-1:y+2,x-1:x+2].ElementsByRow.Sum - a[y,x];

procedure NextGen;
begin
  foreach var (y,x) in (1..m-2).Cartesian(1..n-2) do
    case InhCount(x,y) of
      0..1,4..9: b[y,x] := 0;
      3: b[y,x] := 1; 
      2: b[y,x] := a[y,x]; 
    end;
  a := Copy(b);
  Draw;
end;

begin
  Window.Title := 'Игра Жизнь Джона Конвея';
  Init;
  BeginFrameBasedAnimation(NextGen,5);
end.