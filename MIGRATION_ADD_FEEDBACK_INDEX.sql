-- =============================================
-- MIGRATION: Add Index on Feedback Analysis ID
-- =============================================
-- Run this SQL in your Supabase SQL Editor
-- This improves query performance when looking up feedback by analysis_id

-- Add index on feedback.analysis_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_feedback_analysis_id ON public.feedback(analysis_id);

-- Add index on feedback.user_id for faster user-specific queries
CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON public.feedback(user_id);

-- Add index on feedback.created_at for chronological queries
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON public.feedback(created_at DESC);

-- =============================================
-- VERIFICATION QUERIES
-- =============================================

-- View the relationship between analysis_history and feedback
SELECT
    ah.id as analysis_id,
    ah.analysis_type,
    ah.prediction as predicted,
    ah.created_at as analyzed_at,
    f.id as feedback_id,
    f.actual_label as actual,
    f.feedback_text,
    f.created_at as feedback_at
FROM public.analysis_history ah
LEFT JOIN public.feedback f ON ah.id = f.analysis_id
ORDER BY ah.created_at DESC
LIMIT 10;

-- Count analyses with and without feedback
SELECT
    COUNT(*) as total_analyses,
    COUNT(f.id) as analyses_with_feedback,
    COUNT(*) - COUNT(f.id) as analyses_without_feedback
FROM public.analysis_history ah
LEFT JOIN public.feedback f ON ah.id = f.analysis_id;

-- =============================================
-- SUCCESS MESSAGE
-- =============================================
DO $$
BEGIN
    RAISE NOTICE '✅ Migration completed successfully!';
    RAISE NOTICE '📊 Added indexes on feedback table for better performance';
    RAISE NOTICE '🔗 Relationship: analysis_history.id <-> feedback.analysis_id';
    RAISE NOTICE '';
    RAISE NOTICE '📝 The tables are now optimized for queries that:';
    RAISE NOTICE '   - Look up feedback by analysis_id';
    RAISE NOTICE '   - Find all feedback from a specific user';
    RAISE NOTICE '   - Sort feedback chronologically';
END $$;
