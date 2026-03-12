type
  Farenheit =  class;
  
  Celsius = auto class
    value: real;
  public  
    constructor (v: real) := value := v;
    static function operator implicit(c: Celsius): Farenheit;
    function ToString: string; override := $'{value}°C';
  end;

  Farenheit = auto class
    value: real;
  public  
    constructor (v: real) := value := v;
    static function operator implicit(f: Farenheit): Celsius;
    function ToString: string; override := $'{value}°F';
  end;

static function Celsius.operator implicit(c: Celsius): Farenheit 
  := new Farenheit(c.value * 9 / 5 + 32);

static function Farenheit.operator implicit(f: Farenheit): Celsius 
  := new Celsius((f.value - 32) * 5 / 9);
 
begin
  var c := new Celsius(25);
  var f: Farenheit := c; // Неявное преобразование из Цельсия в Фаренгейт
  Println($'Температура: {c} = {f}');
  
  var f2 := new Farenheit(77);
  var c2: Celsius := f2; // Неявное преобразование из Фаренгейта в Цельсий
  Println($'Температура: {f2} = {c2}');
end.