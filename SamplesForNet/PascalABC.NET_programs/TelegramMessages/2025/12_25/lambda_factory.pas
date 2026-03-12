function less(a: integer): integer -> boolean
  := x -> x < a;

begin
  Arr(1..9).Where(less(5)).Print
end.