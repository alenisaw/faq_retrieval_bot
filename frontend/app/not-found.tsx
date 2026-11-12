import Link from "next/link";

export default function NotFound() {
  return (
    <div className="max-w-[56ch]">
      <h1 className="font-display text-[28px] font-semibold leading-tight sm:text-[36px]">Такой страницы нет</h1>
      <p className="mt-3 text-ink-soft">
        Возможно, ссылка устарела. Задайте вопрос на главной — бот поищет ответ в базе приёмной комиссии.
      </p>
      <Link
        href="/"
        className="mt-6 inline-flex h-11 items-center rounded-lg bg-ultramarine px-5 font-medium text-white hover:bg-[#2537b8]"
      >
        Задать вопрос
      </Link>
    </div>
  );
}
