import builtins

import pytest

import main


def feed(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(it))


def test_ask_choice_by_number_name_default_and_retry(monkeypatch, capsys):
    feed(monkeypatch, ["9", "abc", "2"])
    assert main.ask_choice("Метод", ["a", "b", "c"]) == "b"
    assert "Введите номер" in capsys.readouterr().out
    feed(monkeypatch, ["c"])
    assert main.ask_choice("Метод", ["a", "b", "c"]) == "c"
    feed(monkeypatch, [""])
    assert main.ask_choice("Метод", ["a", "b"], default="a") == "a"


def test_menu_handles_unknown_item_errors_and_exit(monkeypatch, capsys):
    def boom():
        raise FileNotFoundError("нет файла")

    monkeypatch.setitem(main.ACTIONS, "1", boom)
    monkeypatch.setattr(main, "log_action", lambda *a, **k: None)
    feed(monkeypatch, ["42", "1", "0"])
    main.menu()
    out = capsys.readouterr().out
    assert "Нет такого пункта" in out and "Ошибка: нет файла" in out and "До свидания" in out


def test_menu_exits_on_eof(monkeypatch, capsys):
    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", eof)
    main.menu()
    assert "До свидания" in capsys.readouterr().out


def test_log_action_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "MENU_LOG", tmp_path / "log.jsonl")
    main.log_action("test", n=1)
    main.log_action("test", n=2)
    lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and '"n": 2' in lines[1]


@pytest.mark.parametrize("argv", [["--setup", "--serve"], ["--ask"]])
def test_cli_rejects_bad_args(argv):
    with pytest.raises(SystemExit):
        main.main(argv)
