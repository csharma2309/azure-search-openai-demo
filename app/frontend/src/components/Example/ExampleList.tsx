import { Example } from "./Example";
import { useTranslation } from "react-i18next";

import styles from "./Example.module.css";

interface Props {
    onExampleClicked: (value: string) => void;
    useGPT4V?: boolean;
}

export const ExampleList = ({ onExampleClicked, useGPT4V }: Props) => {
    const { t } = useTranslation();

    //const DEFAULT_EXAMPLES: string[] = [t("defaultExamples.1"), t("defaultExamples.2"), t("defaultExamples.3")];
    //const GPT4V_EXAMPLES: string[] = [t("gpt4vExamples.1"), t("gpt4vExamples.2"), t("gpt4vExamples.3")];
    const DEFAULT_EXAMPLES: string[] = [
    "How to improve quality of farm soil specifically for cultivating corn?",
    "What are the benefits of using zinc seed treatments for corn cultivation?",
    "What factors should be considered for optimal Soybean cultivation?"
];

const GPT4V_EXAMPLES: string[] = [
    "How to improve quality of farm soil specifically for cultivating corn?",
    "What are the benefits of using zinc seed treatments for corn cultivation?",
    "What factors should be considered for optimal Soybean cultivation?"
];



    return (
        <ul className={styles.examplesNavList}>
            {(useGPT4V ? GPT4V_EXAMPLES : DEFAULT_EXAMPLES).map((question, i) => (
                <li key={i}>
                    <Example text={question} value={question} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
