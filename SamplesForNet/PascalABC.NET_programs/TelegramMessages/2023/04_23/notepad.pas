uses GraphWPF;

begin
  Font.Size := 40;
  var (x,y) := (0.0, 0.0);
  
  OnKeyPress := c -> begin
    TextOut(x,y,c);
    x += TextWidth(if c <> ' ' then c else 'a');
  end;
end.
