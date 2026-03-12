uses GraphWPF;

function Reshape(data: array of integer; w,h: integer): array [,] of integer;
begin
  if data.Length <> w * h then
    raise new Exception('Несоответствие размерности при Reshape()');   
  Result := MatrByCol(data.Batch(w))
end;

procedure DrawPixels(pixels: array [,] of integer; x: real := 0; y: real := 0; width: real := 10)
  := FastDraw(dc -> begin
    for var ix := 0 to pixels.RowCount-1 do
    for var iy := 0 to pixels.ColCount-1 do
      DrawRectangleDC(dc,x + ix * width, y + iy * width, width, width, GrayColor(pixels[ix,iy]),Colors.Transparent,1);
  end);

begin
  var sz_img := 28;
  var strs: array of string := ReadLines('mnist_test.csv').Skip(1).ToArray;
  var data: array of array of integer := strs.Select(s -> s.ToWords(',').Skip(1).Select(x -> x.ToInteger).ToArray).ToArray;
  var digits := data.Select(arr -> Reshape(arr,sz_img,sz_img)).ToArray;
  
  var w := 2;
  var digits_in_col := 15;
  var digits_in_row := 11;
  for var iy:=0 to digits_in_row-1 do
  for var ix:=0 to digits_in_col-1 do
    DrawPixels(digits[iy * digits_in_col + ix],ix*w*sz_img,iy*w*sz_img,w);
end.