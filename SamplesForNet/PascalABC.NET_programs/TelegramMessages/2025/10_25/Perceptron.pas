function LinearCombination(x, w: array of real; b: real): real;
begin
  Result := b;
  for var i := 0 to x.Length - 1 do
    Result += x[i] * w[i];
end;
  
function Sigmoid(z: real) := 1 / (1 + Exp(-z));

function Predict(x, w: array of real; b: real; activation: real -> real) := 
  activation(LinearCombination(x, w, b));

begin
  var weights := [0.8, -0.3, 0.5, -0.2];
  var bias := 0.1;
  var x := [45.0, 120.0, 5.5, 24.5];

  var probability := Predict(x, weights, bias, Sigmoid);
  Println('Вероятность диабета:', probability);
end.