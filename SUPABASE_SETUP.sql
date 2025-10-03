-- AI Detection System - Supabase Database Setup (Simplified)
-- Run this SQL in your Supabase SQL Editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================
-- PROFILES TABLE (Simplified - No Credits/Subscriptions)
-- =============================================
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    avatar_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================
-- ANALYSIS HISTORY TABLE
-- =============================================
CREATE TABLE IF NOT EXISTS public.analysis_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    text_sample TEXT,
    text_length INTEGER,
    prediction VARCHAR(50) CHECK (prediction IN ('AI', 'Human', 'Mixed')),
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    analysis_type VARCHAR(50) DEFAULT 'basic' CHECK (analysis_type IN ('basic', 'mixed_content', 'fact_check')),
    ai_percentage FLOAT,
    human_percentage FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================
-- FEEDBACK TABLE (Optional - for model improvement)
-- =============================================
CREATE TABLE IF NOT EXISTS public.feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.profiles(id) ON DELETE SET NULL,
    analysis_id UUID REFERENCES public.analysis_history(id) ON DELETE SET NULL,
    text_sample TEXT,
    predicted_label VARCHAR(50),
    actual_label VARCHAR(50),
    feedback_text TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================
-- INDEXES FOR PERFORMANCE
-- =============================================
CREATE INDEX IF NOT EXISTS idx_analysis_user_id ON public.analysis_history(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_created_at ON public.analysis_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_profiles_email ON public.profiles(email);

-- =============================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- =============================================

-- Enable RLS on all tables
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.feedback ENABLE ROW LEVEL SECURITY;

-- Profiles policies
CREATE POLICY "Users can view own profile" ON public.profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.profiles
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile" ON public.profiles
    FOR INSERT WITH CHECK (auth.uid() = id);

-- Analysis history policies
CREATE POLICY "Users can view own analysis history" ON public.analysis_history
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own analysis" ON public.analysis_history
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Feedback policies
CREATE POLICY "Users can view own feedback" ON public.feedback
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own feedback" ON public.feedback
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- =============================================
-- TRIGGERS
-- =============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to profiles table
CREATE TRIGGER update_profiles_updated_at BEFORE UPDATE ON public.profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================
-- FUNCTION: Create Profile on Signup
-- =============================================
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, first_name, last_name, avatar_url)
    VALUES (
        NEW.id,
        NEW.email,
        NEW.raw_user_meta_data->>'first_name',
        NEW.raw_user_meta_data->>'last_name',
        NEW.raw_user_meta_data->>'avatar_url'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to create profile on user signup
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- =============================================
-- FUNCTION: Get User Stats
-- =============================================
CREATE OR REPLACE FUNCTION public.get_user_stats(p_user_id UUID)
RETURNS TABLE (
    total_analyses BIGINT,
    ai_detections BIGINT,
    human_detections BIGINT,
    mixed_detections BIGINT,
    avg_confidence FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT as total_analyses,
        COUNT(*) FILTER (WHERE prediction = 'AI')::BIGINT as ai_detections,
        COUNT(*) FILTER (WHERE prediction = 'Human')::BIGINT as human_detections,
        COUNT(*) FILTER (WHERE prediction = 'Mixed')::BIGINT as mixed_detections,
        AVG(ah.confidence)::FLOAT as avg_confidence
    FROM public.analysis_history ah
    WHERE ah.user_id = p_user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================
-- SUCCESS MESSAGE
-- =============================================
DO $$
BEGIN
    RAISE NOTICE '✅ Database setup completed successfully!';
    RAISE NOTICE '📊 Tables created: profiles, analysis_history, feedback';
    RAISE NOTICE '🔒 Row Level Security enabled on all tables';
    RAISE NOTICE '⚡ Triggers and functions created';
    RAISE NOTICE '';
    RAISE NOTICE '🚀 Your AI Detection System database is ready!';
    RAISE NOTICE '';
    RAISE NOTICE '👤 Users will be greeted as: Welcome {first_name}!';
END $$;
