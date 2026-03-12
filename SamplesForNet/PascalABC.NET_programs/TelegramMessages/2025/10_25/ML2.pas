// Прогноз: ŷ = w1*x1 + w2*x2 + b
function Predict(x1, x2, w1, w2, b: real) := w1 * x1 + w2 * x2 + b;

// Квадратичная ошибка
function Loss(pred, y: real) := Sqr(pred - y);

// Градиенты по параметрам
function d_w1(x1, pred, y: real) := 2 * x1 * (pred - y);

function d_w2(x2, pred, y: real) := 2 * x2 * (pred - y);

function d_b(pred, y: real) := 2 * (pred - y);

begin
  // Данные: [часы подготовки, часы сна] -> оценка
  var x: array of array of real := (
    (2, 6), (4, 5), (5, 7), (1, 4), (3, 6),
    (6, 5), (5, 6), (7, 8), (3, 5), (4, 7));
  var y: array of real := (70, 78, 82, 62, 75, 85, 80, 90, 72, 79);
  var n := x.Length;
  
  // Параметры модели
  var w1 := 0.1;
  var w2 := 0.1;
  var b := 0.1;
  
  var lr := 0.005;      // скорость обучения (чуть больше — быстрее сойдётся)
  var epochs := 20000;   // больше эпох — глубже сходимость
  
  var errors := new real[epochs];
  
  for var epoch := 0 to epochs - 1 do
  begin
    var total_loss := 0.0;
    var grad_w1 := 0.0;
    var grad_w2 := 0.0;
    var grad_b := 0.0;
    
    // аккумулируем сумму по всему датасету (batch)
    for var i := 0 to n - 1 do
    begin
      var p := Predict(x[i][0], x[i][1], w1, w2, b);
      total_loss += Loss(p, y[i]);
      grad_w1 += d_w1(x[i][0], p, y[i]);
      grad_w2 += d_w2(x[i][1], p, y[i]);
      grad_b += d_b(p, y[i]);
    end;
    
    // обновление параметров по среднему градиенту
    w1 -= lr * grad_w1 / n;
    w2 -= lr * grad_w2 / n;
    b -= lr * grad_b / n;
    
    // средняя ошибка на эпохе
    errors[epoch] := total_loss / n;
    
    // раз в 1000 эпох — короткий лог
    if ((epoch + 1) mod 1000 = 0) then
      Println($'Эпоха {epoch+1}: ошибка = {errors[epoch]:0.0000}; w1={w1:0.000}, w2={w2:0.000}, b={b:0.000}');
  end;
  
  Println;
  Println('Проверка модели:');
  var tests: array of array of real := (
    (4, 6),  // средний ученик
    (6, 5),  // хорошо готовился
    (2, 8)   // мало занимался, но хорошо выспался
  );
  
  for var i := 0 to tests.GetLength(0) - 1 do
  begin
    var tx1 := tests[i, 0];
    var tx2 := tests[i, 1];
    var p := Predict(tx1, tx2, w1, w2, b);
    Println($'Подготовка={tx1} ч, Сон={tx2} ч → прогноз: {Round(p, 2)} баллов');
  end
  
end.