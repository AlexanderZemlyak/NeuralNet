uses GraphWPF;

begin
  Window.Title := 'Приветствие участникам IV конференции PascalABC.NET';
  Pen.Color := Colors.Blue;
  Pen.Width := 3;
  OnMouseDown := (x,y,mb) → 
    MoveTo(x,y);
  OnMouseMove := (x,y,mb) → 
    if mb=1 then LineTo(x,y);
  OnKeyDown := k → 
    if k = Key.Space then Window.Clear;
end.
