##
// +∞: результат деления на 0 или константа PositiveInfinity
var posInf := real.PositiveInfinity;
var posInf2 := 1.0/0.0;
Println($'+∞: {posInf}  Проверка: {real.IsPositiveInfinity(posInf)}');
Println($'1/0 → {posInf2}');

// -∞: отрицательное деление на 0 или константа NegativeInfinity
var negInf := real.NegativeInfinity;
var negInf2 := -1.0/0.0;
Println($'-∞: {negInf}  Проверка: {real.IsNegativeInfinity(negInf)}');
Println($'-1/0 → {negInf2}');

// NaN: «не число» – результат недопустимых операций
var nanVal := real.NaN;
var nanVal2 := Sqrt(-1);
Println($'NaN: {nanVal}  Проверка: {real.IsNaN(nanVal)}');
Println($'Sqrt(-1) → {nanVal2}');
