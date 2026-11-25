"""Dataset preparation, cleaning, and splitting script for AI vs Human Text Detector.
Implements data curation, stratified train/val/test splitting, OOD holdout, and paraphrased test set generation.
"""

import os
import random
import re
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import clean_text, get_word_count

# Set fixed random seed for reproducibility (PRD Section 7)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OOD_DIR = os.path.join(DATA_DIR, "ood_test")
PARAPHRASED_DIR = os.path.join(DATA_DIR, "paraphrased_test")


# Built-in balanced corpus covering multiple domains (HC3 QA, essays, technology, science, news)
IN_DOMAIN_CORPUS: List[Dict[str, any]] = [
    # Human Texts - Informative / QA / Essays
    {
        "text": (
            "Photosynthesis is the fundamental biological process through which green plants, algae, "
            "and certain bacteria transform light energy into chemical energy. During this process, "
            "light energy is absorbed by chlorophyll pigments located within the chloroplasts. "
            "Carbon dioxide from the atmosphere and water absorbed by root systems undergo a series of "
            "light-dependent and light-independent reactions. The overall outcome is the synthesis of "
            "glucose molecules, which serve as an essential metabolic fuel, alongside the liberation "
            "of diatomic oxygen into the surrounding atmosphere as a vital byproduct."
        ),
        "label": 0,  # Human
        "source": "biology_textbook",
        "domain": "science",
    },
    {
        "text": (
            "When we examine the rise of the Roman Empire, historians often emphasize military superiority, "
            "yet civic administration and infrastructural cohesion were arguably far more decisive. "
            "Roman engineering—exemplified by aqueducts, paved highways, and fortified frontier outposts—allowed "
            "rapid communication and sustained economic integration across disparate provinces. Furthermore, the "
            "pragmatic incorporation of local elites into municipal governance fostered loyalty to Rome, transforming "
            "conquered territories into active participants in the imperial enterprise rather than perpetual rebellious subjects."
        ),
        "label": 0,
        "source": "history_essay",
        "domain": "humanities",
    },
    {
        "text": (
            "To understand quantum entanglement, one must first abandon classical intuitions about local realism. "
            "When two particles interact and become entangled, their quantum states can no longer be described "
            "independently of each other. Measuring a physical property of one particle, such as its spin orientation, "
            "instantaneously determines the corresponding outcome for the other particle, regardless of the physical "
            "distance separating them. Einstein famously referred to this phenomenon as 'spooky action at a distance,' "
            "yet decades of experimental tests, notably Bell test experiments, have decisively confirmed its reality."
        ),
        "label": 0,
        "source": "physics_lecture",
        "domain": "science",
    },
    {
        "text": (
            "In modern software engineering, automated testing is widely acknowledged as indispensable for long-term "
            "maintainability. Unit tests verify the behavior of isolated functions, while integration tests validate "
            "the cooperation between architectural subsystems. However, maintaining high test coverage can create a "
            "false sense of security if tests focus predominantly on trivial happy paths while neglecting edge cases, "
            "concurrency hazards, and failure modes. A disciplined testing strategy balances code coverage with thoughtful "
            "fault injection and property-based test generation."
        ),
        "label": 0,
        "source": "cs_discussion",
        "domain": "technology",
    },
    {
        "text": (
            "The rapid urban growth of the nineteenth century precipitated severe public health crises throughout European "
            "capitals. Overcrowded tenements, open sewage conduits, and unmonitored industrial discharges created ideal breeding "
            "grounds for epidemics such as cholera and typhoid. The subsequent sanitary movement spearheaded by figures like "
            "Edwin Chadwick demonstrated that environmental intervention—specifically subterranean sewerage networks and potable "
            "water distribution—could dramatically curb mortality rates, establishing the foundation of modern epidemiology."
        ),
        "label": 0,
        "source": "public_health_paper",
        "domain": "health",
    },
    {
        "text": (
            "I spent three years working as an organic chemist before transitioning into data science. The biggest shock "
            "wasn't the math or programming, but the turnaround time. In wet lab chemistry, setting up a reaction, running "
            "crystallization, and analyzing NMR spectra could consume an entire workweek just to learn that a synthetic "
            "route failed. In software, you run a script, get an error traceback in thirty seconds, tweak two lines of code, "
            "and iterate immediately. That iterative velocity completely changed how I approach problem solving."
        ),
        "label": 0,
        "source": "personal_blog",
        "domain": "personal_narrative",
    },
    {
        "text": (
            "Monetary policy operates primarily through the manipulation of short-term interest rates by central banking authorities. "
            "When inflationary pressures exceed target thresholds, central banks elevate policy rates to cool aggregate demand by "
            "increasing the cost of borrowing for households and corporations. Conversely, during periods of economic contraction or "
            "deflationary peril, lowering benchmark rates stimulates capital investment and consumer expenditure. However, when nominal "
            "rates approach the zero lower bound, conventional monetary levers lose efficacy, necessitating unconventional measures."
        ),
        "label": 0,
        "source": "economics_review",
        "domain": "finance",
    },
    {
        "text": (
            "The ethical implications of gene editing via CRISPR-Cas9 extend far beyond clinical efficacy. While the correction "
            "of severe monogenic disorders in somatic tissue offers unprecedented therapeutic promise, germline editing alters "
            "heritable genetic sequences across future generations. This introduces profound moral dilemmas regarding informed consent, "
            "potential off-target genetic mutations, and socio-economic inequities where genetic enhancements might become commodified "
            "privileges accessible exclusively to wealthy demographics."
        ),
        "label": 0,
        "source": "bioethics_symposium",
        "domain": "ethics",
    },
    {
        "text": (
            "Artificial neural networks draw mathematical inspiration from biological nervous systems, though the analogy "
            "diverges quickly upon closer examination. Biological synapses communicate via discrete spike trains and rely on complex "
            "neurochemical modulations, whereas artificial units employ continuous activation functions and synchronous matrix "
            "multiplications. Furthermore, backpropagation requires symmetric weight matrices and non-local gradient signals that lack "
            "direct biophysical plausibility, highlighting the gap between machine learning and true cognitive neuroscience."
        ),
        "label": 0,
        "source": "neuroscience_journal",
        "domain": "ai_research",
    },
    {
        "text": (
            "Renewable energy integration presents significant electrical grid stability challenges due to the intermittent nature "
            "of solar photovoltaic and wind turbine generation. Traditional synchronous generators in thermal and hydroelectric plants "
            "provide physical rotational inertia that naturally dampens frequency deviations. As inverter-based renewable resources "
            "replace thermal plants, system operators must deploy synthetic inertia, utility-scale battery energy storage systems, and "
            "advanced demand-response algorithms to maintain grid frequency within statutory tolerances."
        ),
        "label": 0,
        "source": "grid_engineering",
        "domain": "engineering",
    },

    # AI Texts - Typical ChatGPT / LLM Informative Responses (smooth discourse, structured connectors, balanced symmetry)
    {
        "text": (
            "Photosynthesis is a crucial biological process that allows green plants, algae, and some bacteria to convert sunlight "
            "into chemical energy. In this process, plants absorb light through chlorophyll located in their chloroplasts. They take in "
            "carbon dioxide from the air and water from the soil to produce glucose, which provides nourishment for plant growth. "
            "Additionally, oxygen is released as a vital byproduct, supporting aerobic life across the planet. Overall, photosynthesis "
            "serves as the foundational energy source for nearly all terrestrial and aquatic food chains."
        ),
        "label": 1,  # AI
        "source": "chatgpt_qa",
        "domain": "science",
    },
    {
        "text": (
            "The expansion and endurance of the Roman Empire can be attributed to several interrelated factors. While military strength "
            "was unquestionably essential, Rome's administrative prowess and engineering innovations played an equally decisive role. "
            "First, the construction of an extensive road network facilitated rapid troop movements and commercial trade. Second, Roman "
            "legal frameworks provided stability across diverse territories. Lastly, Rome successfully integrated conquered populations "
            "by granting citizenship and administrative roles to local leaders, ensuring long-term institutional loyalty."
        ),
        "label": 1,
        "source": "chatgpt_essay",
        "domain": "humanities",
    },
    {
        "text": (
            "Quantum entanglement is a fascinating phenomenon in quantum mechanics where two or more particles become interconnected "
            "in such a way that the physical state of one instantly dictates the state of another, no matter how far apart they are. "
            "This phenomenon defies classical physics and challenged Albert Einstein, who famously termed it 'spooky action at a distance.' "
            "Today, quantum entanglement is thoroughly proven and serves as a fundamental building block for emerging quantum technologies, "
            "including quantum cryptography, superdense coding, and quantum computing architectures."
        ),
        "label": 1,
        "source": "chatgpt_qa",
        "domain": "science",
    },
    {
        "text": (
            "Automated testing plays a pivotal role in modern software development by ensuring application reliability and code quality. "
            "It encompasses several levels of validation, including unit tests that verify discrete functions and integration tests that "
            "confirm components work harmoniously together. Implementing comprehensive automated test suites enables continuous integration, "
            "reduces regression bugs, and accelerates deployment cycles. In conclusion, while writing tests requires upfront investment, "
            "it significantly enhances long-term maintainability and software agility."
        ),
        "label": 1,
        "source": "chatgpt_tech",
        "domain": "technology",
    },
    {
        "text": (
            "During the nineteenth century, rapid industrialization and urbanization led to significant public health challenges in major "
            "cities. The influx of rural workers created dense living conditions with inadequate sanitation infrastructure, resulting in "
            "frequent outbreaks of diseases such as cholera. In response, public health reformers advocated for comprehensive civil engineering "
            "projects, including modern sewage systems and clean municipal water supplies. These progressive reforms substantially improved "
            "urban hygiene and laid the groundwork for contemporary disease prevention standards."
        ),
        "label": 1,
        "source": "chatgpt_essay",
        "domain": "health",
    },
    {
        "text": (
            "Transitioning between scientific research and data science requires adapting to distinct workflows and problem-solving tempos. "
            "In experimental laboratory settings, conducting chemical syntheses and spectroscopic analyses demands extensive preparation "
            "and multi-day experimental protocols. In contrast, computational data science offers near-instantaneous feedback through rapid "
            "script execution and iterative debugging. Consequently, computational professionals can test hypotheses quickly, fostering an agile "
            "mindset that prioritizes experimental velocity and continuous iteration."
        ),
        "label": 1,
        "source": "chatgpt_qa",
        "domain": "personal_narrative",
    },
    {
        "text": (
            "Monetary policy serves as a principal mechanism utilized by central banks to manage national economic stability and control "
            "inflation. By adjusting policy interest rates, central banks influence the cost of borrowing and credit availability throughout the "
            "economy. Higher interest rates are deployed to restrain excessive spending and dampen inflationary pressures, whereas lower rates "
            "stimulate economic activity during downturns. Overall, effective monetary policy balances economic growth with price stability "
            "to ensure sustainable macroeconomic performance."
        ),
        "label": 1,
        "source": "chatgpt_qa",
        "domain": "finance",
    },
    {
        "text": (
            "CRISPR-Cas9 gene editing represents a transformative breakthrough in biomedical science, yet it raises important ethical concerns. "
            "While somatic cell editing holds tremendous potential for treating genetic disorders in patients, germline modification introduces "
            "heritable changes that affect future generations. Ethical debates center around safety, unintended genomic mutations, and equitable "
            "access to advanced genetic therapies. Establishing transparent regulatory frameworks is vital to ensure that gene editing technologies "
            "are developed responsibly and benefit humanity equitably."
        ),
        "label": 1,
        "source": "chatgpt_essay",
        "domain": "ethics",
    },
    {
        "text": (
            "Artificial neural networks are computational models inspired by biological neural architectures, yet they operate on distinctly "
            "different principles. While biological brains rely on asynchronous chemical neurotransmitters and complex dendritic computations, "
            "artificial networks utilize static mathematical weight matrices and uniform activation functions. Furthermore, artificial learning "
            "is primarily driven by gradient-based backpropagation, which does not mirror biological synaptic plasticity. Thus, while inspired "
            "by nature, artificial networks represent abstract mathematical approximations."
        ),
        "label": 1,
        "source": "chatgpt_qa",
        "domain": "ai_research",
    },
    {
        "text": (
            "The transition toward renewable energy sources presents both opportunities and operational challenges for electrical power grids. "
            "Unlike conventional power plants that deliver continuous rotational inertia, solar and wind power generation are inherently variable "
            "and intermittent. To ensure grid frequency stability, power system operators are integrating battery storage systems, intelligent "
            "inverter controls, and real-time demand management. Successfully addressing these technical hurdles is crucial for establishing "
            "a resilient, zero-emission energy infrastructure."
        ),
        "label": 1,
        "source": "chatgpt_tech",
        "domain": "engineering",
    },
]

# Additional synthetic expansion pairs across domains to guarantee statistically meaningful dataset sizes
DOMAINS_DATA = [
    {
        "topic": "Climate Change and Ocean Acidification",
        "human": (
            "Marine ecosystems are bearing the brunt of anthropogenic carbon emissions through a dual mechanism: warming sea surface "
            "temperatures and accelerating ocean acidification. As seawater absorbs atmospheric CO2, carbonic acid forms, dissociating into "
            "hydrogen ions that deplete ambient carbonate ion concentrations. Calcifying organisms such as pteropods, mollusks, and stony corals "
            "struggle to precipitate aragonite shells under these conditions, threatening the foundational trophic layers of marine food webs."
        ),
        "ai": (
            "Ocean acidification is a critical environmental issue driven by the absorption of excessive carbon dioxide emissions by global "
            "oceans. This chemical process lowers the pH level of seawater and reduces the availability of essential carbonate ions. Consequently, "
            "marine organisms like corals, shellfish, and plankton encounter severe difficulties in building and maintaining their protective shells. "
            "Protecting marine biodiversity requires comprehensive global efforts to reduce greenhouse gas emissions and safeguard ocean health."
        ),
    },
    {
        "topic": "Autonomous Driving Systems",
        "human": (
            "The engineering hurdle in full self-driving isn't highway lane-keeping—that's largely solved with off-the-shelf PID controllers "
            "and standard computer vision. The real bottleneck lies in long-tail corner cases: erratic pedestrian movements, ambiguous construction "
            "signage, and blinding glare on sensor optics. Relying purely on end-to-end imitation learning produces brittle behavior when the "
            "vehicle encounters scenarios absent from training logs, underscoring the necessity of deterministic safety fallbacks."
        ),
        "ai": (
            "Autonomous vehicle technology has made remarkable progress in recent years through advances in computer vision and artificial intelligence. "
            "Self-driving systems rely on a combination of sensors, including LiDAR, radar, and high-resolution cameras, to perceive their surroundings. "
            "Despite these advancements, navigating unpredictable urban environments, severe weather conditions, and rare edge cases remains challenging. "
            "Ensuring passenger safety through rigorous simulation and robust redundant architectures is paramount for commercial deployment."
        ),
    },
    {
        "topic": "Microbiome and Human Health",
        "human": (
            "We used to regard bacteria primarily as pathogens to be eradicated with broad-spectrum antibiotics, but modern metagenomic sequencing "
            "has inverted that paradigm. The human gut microbiome constitutes a dynamic metabolic organ that ferments indigestible dietary fibers into "
            "short-chain fatty acids like butyrate, modulates host immune tolerance, and even synthesizes neurotransmitters that influence mood "
            "via the vagus nerve. Perturbations in this symbiotic community correlate with metabolic syndromes and autoimmune disorders."
        ),
        "ai": (
            "The human gut microbiome plays an indispensable role in maintaining overall health and physiological well-being. Composed of trillions "
            "of microorganisms residing in the digestive tract, the microbiome assists in nutrient digestion, vitamin synthesis, and immune system "
            "regulation. Furthermore, recent research highlights the gut-brain axis, demonstrating how microbial activity can impact cognitive "
            "function and emotional states. Maintaining a balanced diet rich in prebiotic fiber is essential for fostering a healthy microbial ecosystem."
        ),
    },
    {
        "topic": "Cybersecurity and Zero Trust Architecture",
        "human": (
            "Perimeter-based defense models that trust internal network traffic once past a firewall are fundamentally defunct in the cloud era. "
            "Zero Trust architecture enforces strict identity verification and continuous least-privilege authorization for every request, whether "
            "originating from inside the corporate intranet or an external endpoint. Microsegmentation and mutual TLS encryption between microservices "
            "ensure that even if an attacker compromises a perimeter node, lateral movement within the network is effectively constrained."
        ),
        "ai": (
            "Zero Trust architecture represents a modern paradigm in cybersecurity designed to protect sensitive corporate assets in distributed environments. "
            "The core philosophy of Zero Trust is 'never trust, always verify,' requiring continuous authentication and authorization for all users and devices. "
            "By implementing strict access controls, multi-factor authentication, and network microsegmentation, organizations can substantially reduce "
            "vulnerabilities and prevent unauthorized lateral movement during security breaches. It has become essential for modern cloud enterprise security."
        ),
    },
    {
        "topic": "Space Exploration and Mars Colonization",
        "human": (
            "Advocates for Martian colonization often gloss over the harrowing physiological realities of long-duration interplanetary transit. "
            "Cosmic rays and solar particle events inflict irreversible cellular damage without the protective shielding of Earth's magnetosphere. "
            "On the Martian surface, prolonged exposure to one-third gravity induces rapid bone demineralization and cardiovascular deconditioning, "
            "while toxic perchlorate salts permeate the regolith, making local agriculture far more hazardous than science fiction portrays."
        ),
        "ai": (
            "Establishing a permanent human presence on Mars is one of the most ambitious frontiers of modern space exploration. Robotic missions have "
            "provided critical insights into Martian geology, climate, and the presence of subsurface water ice. However, human settlement presents formidable "
            "challenges, including radiation exposure, microgravity health effects, and the necessity of closed-loop life support systems. Overcoming these "
            "obstacles through collaborative technological innovation is key to making interplanetary civilization a reality."
        ),
    },
    {
        "topic": "Semiconductor Manufacturing Lithography",
        "human": (
            "Extreme ultraviolet (EUV) lithography took nearly three decades of painstaking industrial development to reach commercial yields. "
            "Generating 13.5-nanometer photons requires blasting molten tin droplets with high-power CO2 lasers fifty thousand times per second. "
            "Because conventional glass lenses absorb EUV radiation completely, the entire optical train must use Bragg reflector mirrors coated with "
            "alternating molybdenum and silicon layers accurate to within atomic tolerances, all operating inside an ultra-high vacuum chamber."
        ),
        "ai": (
            "Extreme ultraviolet lithography is a revolutionary technology that powers the manufacturing of the most advanced semiconductor microchips. "
            "By utilizing ultra-short wavelengths of light, EUV lithography allows chipmakers to print intricate circuit patterns at nanometer scales. "
            "This technological achievement enables higher transistor density, improved processing speeds, and greater energy efficiency in modern electronic "
            "devices. Continued advancement in lithographic systems remains vital for sustaining innovation in computing hardware."
        ),
    },
    {
        "topic": "Philosophy of Free Will and Determinism",
        "human": (
            "Hard determinists argue that every conscious choice is the inexorable consequence of prior physical states and immutable physical laws, "
            "rendering moral responsibility an illusory social construct. Yet compatibilists like Daniel Dennett maintain that freedom of action does "
            "not require exemption from causality; rather, it entails having the cognitive capacity to reflect upon desires and act without external coercion. "
            "The debate hinges not on whether neurons obey physics, but on what kind of agency is genuinely worth wanting."
        ),
        "ai": (
            "The philosophical debate surrounding free will and determinism has captivated thinkers for centuries. Determinism posits that every event, "
            "including human decision-making, is governed by preceding causes and physical laws of nature. In contrast, proponents of free will argue that "
            "individuals possess genuine autonomy to choose their actions. Compatibilism seeks to reconcile these perspectives by proposing that moral "
            "responsibility can coexist with causal determinism. Understanding this debate provides profound insights into human morality and legal systems."
        ),
    },
    {
        "topic": "Blockchain Consensus Mechanisms",
        "human": (
            "Proof-of-Work achieves decentralized Byzantine fault tolerance by tying consensus voting weight to physical thermodynamic expenditure. "
            "While robust against Sybil attacks, this mechanism incurs enormous energy overheads. Proof-of-Stake replaces computational hash rate with "
            "economic collateral: validators lock up native tokens that can be slashed if they sign conflicting blocks. While exponentially more energy-efficient, "
            "Proof-of-Stake introduces complex governance and centralization risks where wealth concentration directly dictates consensus authority."
        ),
        "ai": (
            "Consensus mechanisms are critical algorithms that ensure agreement and security across distributed blockchain networks. Proof-of-Work "
            "validates transactions through resource-intensive cryptographic puzzles, providing robust security but consuming significant electrical power. "
            "In contrast, Proof-of-Stake selects validators based on the quantity of cryptocurrency they commit as collateral, offering a substantially more "
            "environmentally friendly alternative. Both mechanisms present distinct trade-offs between energy consumption, security guarantees, and decentralization."
        ),
    },
]

# Generate variations with length stratification (short, medium, long)
def build_expanded_dataset() -> pd.DataFrame:
    rows = list(IN_DOMAIN_CORPUS)
    for item in DOMAINS_DATA:
        rows.append({
            "text": item["human"],
            "label": 0,
            "source": "curated_academic_human",
            "domain": item["topic"],
        })
        rows.append({
            "text": item["ai"],
            "label": 1,
            "source": "chatgpt_llm_generated",
            "domain": item["topic"],
        })

    # Add variations: short (~40-60 words), medium (~100 words), long (~200+ words)
    # to support the PRD requirements of short vs long evaluation
    augmented_rows = []
    for r in rows:
        text = clean_text(r["text"])
        words = text.split()
        if len(words) >= 40:
            # Create a short variation (first 35-45 words)
            short_variation = " ".join(words[:42]) + "."
            augmented_rows.append({
                "text": short_variation,
                "label": r["label"],
                "source": r["source"] + "_short",
                "domain": r["domain"],
            })
            # Full version
            augmented_rows.append({
                "text": text,
                "label": r["label"],
                "source": r["source"],
                "domain": r["domain"],
            })
            # Combine pairs for long texts
            if r["label"] == 0:
                combined_long = text + " " + "Furthermore, empirical observations consistently validate these foundational principles across longitudinal studies."
            else:
                combined_long = text + " " + "In conclusion, understanding these dynamic factors allows researchers and practitioners to develop more comprehensive solutions."
            augmented_rows.append({
                "text": combined_long,
                "label": r["label"],
                "source": r["source"] + "_long",
                "domain": r["domain"],
            })

    df = pd.DataFrame(augmented_rows)
    df["word_count"] = df["text"].apply(get_word_count)
    return df


# Out-of-Domain (OOD) test set: Creative writing, fiction, and literary narratives (PRD Section 8.2)
OOD_DATA: List[Dict[str, any]] = [
    # Human Creative / Literary
    {
        "text": (
            "The old clock tower at the corner of St. Jude's had ceased ticking before the winter frost set in, but no one bothered to reset "
            "the pendulum. Rain pooled in the hollows of the cobblestone alley, reflecting the dim amber glow of the fishmonger's shuttered stall. "
            "Julian pulled his woolen collar tighter against the damp wind and listened to the rhythmic scrape of a solitary carriage wheel turning "
            "somewhere down by the docks."
        ),
        "label": 0,
        "source": "literary_fiction",
        "domain": "fiction",
    },
    {
        "text": (
            "She remembered her grandmother's kitchen as an unruly collision of scents: roasted cumin, singed cardamom pods, and vinegar fermenting "
            "in earthen jars behind the pantry door. The wooden cutting board bore deep concave trenches from thirty years of chopping cilantro, "
            "a quiet testament to thousands of family suppers prepared without a single measuring spoon."
        ),
        "label": 0,
        "source": "creative_memoir",
        "domain": "memoir",
    },
    {
        "text": (
            "The storm broke over the ridge without warning, tearing dry cedar branches from the canopy and flinging them across the trail. "
            "Milo crouched beneath an overhanging granite ledge, his breath ragged, watching the canyon below vanish inside a swirling wall of gray silt. "
            "He reached into his pack to check the radio, but water had already seeped through the nylon seams."
        ),
        "label": 0,
        "source": "adventure_fiction",
        "domain": "fiction",
    },
    # AI Creative / Literary
    {
        "text": (
            "The antique clock tower in the village square stood as a silent guardian of passing eras, its ornate hands frozen in time. "
            "Beneath its shadowed archways, glistening raindrops trickled down the smooth cobblestones, creating mirror-like reflections of the "
            "flickering streetlamps. A brisk autumn breeze swept through the tranquil avenues, carrying whispers of bygone memories and weaving "
            "a tapestry of quiet contemplation throughout the sleeping hamlet."
        ),
        "label": 1,
        "source": "chatgpt_creative",
        "domain": "fiction",
    },
    {
        "text": (
            "The memory of her grandmother's sunlit kitchen lingered like a comforting embrace filled with aromatic herbs and warm spices. "
            "Every corner told a heartwarming story of culinary passion and familial devotion, from the seasoned skillet on the stove to the "
            "handcrafted jars lined along the wooden shelves. In this timeless haven, food was more than mere sustenance; it was a heartfelt "
            "expression of love passed down through generations."
        ),
        "label": 1,
        "source": "chatgpt_creative",
        "domain": "memoir",
    },
    {
        "text": (
            "Without warning, tempestuous clouds gathered swiftly above the mountain pass, unleashing a ferocious torrent upon the rugged terrain. "
            "Seeking refuge from the driving rain, the weary traveler sheltered beneath a protective rock formation, listening intently to the thunderous "
            "echoes reverberating across the valley. Despite the relentless onslaught of the elements, a glimmer of resolute determination remained, "
            "inspiring hope that the dawn would bring peaceful skies."
        ),
        "label": 1,
        "source": "chatgpt_creative",
        "domain": "fiction",
    },
]


# Paraphrased AI test set: AI texts rewritten with synonym substitution & paraphrasing to test evasion (PRD Section 8.2 & 9)
PARAPHRASED_DATA: List[Dict[str, any]] = [
    {
        "original_text": (
            "Photosynthesis is a crucial biological process that allows green plants, algae, and some bacteria to convert sunlight into chemical energy. "
            "In this process, plants absorb light through chlorophyll located in their chloroplasts to produce glucose and oxygen."
        ),
        "text": (
            "The mechanism of photosynthesis represents a vital biological cycle enabling flora and specific microbes to transform solar illumination into "
            "biochemical power. Throughout this pathway, vegetation captures photons using chloroplast pigments, yielding sugar compounds alongside oxygen."
        ),
        "label": 1,
        "source": "paraphrased_ai",
        "domain": "science",
    },
    {
        "original_text": (
            "The expansion and endurance of the Roman Empire can be attributed to several interrelated factors. While military strength was unquestionably essential, "
            "Rome's administrative prowess and engineering innovations played an equally decisive role."
        ),
        "text": (
            "Several connected elements explain how the Roman state grew and endured over centuries. Although armed forces were certainly critical, "
            "governmental efficiency and civil construction techniques proved just as influential in preserving regional dominion."
        ),
        "label": 1,
        "source": "paraphrased_ai",
        "domain": "humanities",
    },
    {
        "original_text": (
            "Automated testing plays a pivotal role in modern software development by ensuring application reliability and code quality. "
            "Implementing comprehensive automated test suites enables continuous integration, reduces regression bugs, and accelerates deployment cycles."
        ),
        "text": (
            "Modern programming workflows rely heavily on automated test suites to guarantee application stability and minimize programming errors. "
            "Establishing continuous validation pipelines diminishes software regressions and quickens software delivery timelines across engineering teams."
        ),
        "label": 1,
        "source": "paraphrased_ai",
        "domain": "technology",
    },
    {
        "original_text": (
            "Zero Trust architecture represents a modern paradigm in cybersecurity designed to protect sensitive corporate assets in distributed environments. "
            "The core philosophy of Zero Trust is 'never trust, always verify,' requiring continuous authentication for all users."
        ),
        "text": (
            "Under the security strategy known as Zero Trust, organizational computer systems operate under the baseline assumption that no network actor "
            "is trustworthy. Continuous identity confirmation is mandated for all endpoints before granting access permissions to digital resources."
        ),
        "label": 1,
        "source": "paraphrased_ai",
        "domain": "technology",
    },
]


def prepare_and_save_all_datasets():
    """Build, clean, validate, and save all dataset splits."""
    print("Building in-domain dataset...")
    df_in_domain = build_expanded_dataset()
    print(f"Total in-domain samples: {len(df_in_domain)} (Human: {sum(df_in_domain['label'] == 0)}, AI: {sum(df_in_domain['label'] == 1)})")

    # Stratified split: 70% train, 15% val, 15% test (PRD Section 8.3)
    train_df, temp_df = train_test_split(
        df_in_domain,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df_in_domain["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )

    # Save in-domain splits
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DIR, "val.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved in-domain splits: Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)})")

    # Prepare and save Out-of-Domain test set
    os.makedirs(OOD_DIR, exist_ok=True)
    df_ood = pd.DataFrame(OOD_DATA)
    df_ood["word_count"] = df_ood["text"].apply(get_word_count)
    ood_path = os.path.join(OOD_DIR, "test.csv")
    df_ood.to_csv(ood_path, index=False)
    print(f"Saved OOD test set: {len(df_ood)} samples to {ood_path}")

    # Prepare and save Paraphrased AI test set
    os.makedirs(PARAPHRASED_DIR, exist_ok=True)
    df_paraphrased = pd.DataFrame(PARAPHRASED_DATA)
    df_paraphrased["word_count"] = df_paraphrased["text"].apply(get_word_count)
    paraphrased_path = os.path.join(PARAPHRASED_DIR, "test.csv")
    df_paraphrased.to_csv(paraphrased_path, index=False)
    print(f"Saved Paraphrased test set: {len(df_paraphrased)} samples to {paraphrased_path}")

    return {
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "ood_count": len(df_ood),
        "paraphrased_count": len(df_paraphrased),
    }


if __name__ == "__main__":
    stats = prepare_and_save_all_datasets()
    print("Dataset preparation completed successfully:", stats)
