// Прогноз
function Predict(x1, x2, w1, w2, b: real) := w1*x1 + w2*x2 + b;

// Квадратичная ошибка
function Loss(pred, y: real) := Sqr(pred - y);

// Градиенты
function d_w1(x1, pred, y: real) := 2 * x1 * (pred - y);
function d_w2(x2, pred, y: real) := 2 * x2 * (pred - y);
function d_b(pred, y: real) := 2 * (pred - y);

begin
  // [подготовка, сон] -> оценка
  var x: array of array of real := (
    (2,6),(4,5),(5,7),(1,4),(3,6),
    (6,5),(5,6),(7,8),(3,5),(4,7)
  );
  var y: array of real := (70,78,82,62,75,85,80,90,72,79);
  
  var n := x.Length;

  var w1 := 0.1; var w2 := 0.1; var b := 0.1;
  var lr := 0.002;
  var epochs := 20000;
  var errors := new real[epochs];

  for var epoch := 0 to epochs-1 do
  begin
    var total_loss := 0.0;
    var grad_w1 := 0.0; var grad_w2 := 0.0; var grad_b := 0.0;

    for var i := 0 to n-1 do
    begin
      var p := Predict(x[i][0], x[i][1], w1, w2, b);
      total_loss += Loss(p, y[i]);
      grad_w1 += d_w1(x[i][0], p, y[i]);
      grad_w2 += d_w2(x[i][1], p, y[i]);
      grad_b  += d_b(p, y[i]);
    end;

    w1 -= lr * grad_w1 / n;
    w2 -= lr * grad_w2 / n;
    b  -= lr * grad_b / n;
    errors[epoch] := total_loss / n;

    if ((epoch+1) mod 1000 = 0) then
      Println($'Эпоха {epoch+1}: ошибка={errors[epoch]:0.0000}; w1={w1:0.000}; w2={w2:0.000}; b={b:0.000}');
  end;
end.
