import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LMFAO Augmentation Demo",
  description:
    "Upload one robot-demonstration clip and watch it fan out into a grid of augmented variants, each running a single augmentation live in your browser.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
