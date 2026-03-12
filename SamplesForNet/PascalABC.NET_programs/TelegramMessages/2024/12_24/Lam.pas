begin
  var pr: (real,real) -> (real,real)
    := (x,y) -> (if x < y then (x,y) else (y,x));
  pr(3,2).Print;
end.