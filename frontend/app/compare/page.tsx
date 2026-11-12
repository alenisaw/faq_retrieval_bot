import type { Metadata } from "next";
import { CompareView } from "@/components/compare/compare-view";

export const metadata: Metadata = { title: "Сравнить методы — AITU FAQ" };

export default function Page() {
  return <CompareView />;
}
